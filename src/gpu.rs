//! Vulkan context: instance, ray tracing capable device, memory, buffers, images, swapchain.

use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc};
use gpu_allocator::MemoryLocation;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::ffi::{c_char, c_void, CStr};
use std::mem::ManuallyDrop;
use std::sync::Mutex;

pub struct Buffer {
    pub raw: vk::Buffer,
    pub alloc: Option<Allocation>,
    pub size: u64,
    pub addr: u64,
}

impl Buffer {
    pub fn mapped(&mut self) -> &mut [u8] {
        self.alloc
            .as_mut()
            .and_then(|a| a.mapped_slice_mut())
            .expect("buffer is not host visible")
    }
}

pub struct Image {
    pub raw: vk::Image,
    pub alloc: Option<Allocation>,
    pub view: vk::ImageView,
    pub mip_views: Vec<vk::ImageView>,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    pub mips: u32,
}

pub struct RtProps {
    pub handle_size: u32,
    pub handle_alignment: u32,
    pub base_alignment: u32,
    pub scratch_alignment: u64,
}

pub struct Swapchain {
    pub raw: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub done: Vec<vk::Semaphore>,
}

pub struct Gpu {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    pub surface_fn: ash::khr::surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub pdev: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub swapchain_fn: ash::khr::swapchain::Device,
    pub accel_fn: ash::khr::acceleration_structure::Device,
    pub rt_fn: ash::khr::ray_tracing_pipeline::Device,
    pub rt: RtProps,
    pub allocator: ManuallyDrop<Mutex<Allocator>>,
    pub cmd_pool: vk::CommandPool,
    pub device_name: String,
    pub timestamp_period: f32,
}

unsafe extern "system" fn debug_cb(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _ty: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut c_void,
) -> vk::Bool32 {
    let msg = CStr::from_ptr((*data).p_message).to_string_lossy();
    eprintln!("[vulkan {:?}] {}", severity, msg);
    vk::FALSE
}

pub fn align_up(v: u64, a: u64) -> u64 {
    (v + a - 1) / a * a
}

impl Gpu {
    pub fn new(display: RawDisplayHandle, window: RawWindowHandle, validation: bool) -> Result<Self> {
        unsafe {
            let entry = ash::Entry::load().context("loading vulkan-1.dll")?;
            let mut exts: Vec<*const c_char> = ash_window::enumerate_required_extensions(display)?.to_vec();
            let mut layers: Vec<*const c_char> = vec![];
            // Extensions NVIDIA NGX (DLSS Ray Reconstruction) needs, where the driver offers them.
            let (ngx_inst, ngx_dev) = crate::dlss::required_extensions();
            let avail_inst = entry.enumerate_instance_extension_properties(None).unwrap_or_default();
            for e in &ngx_inst {
                let have = avail_inst.iter().any(|a| a.extension_name_as_c_str().map(|n| n == e.as_c_str()).unwrap_or(false));
                if have && !exts.iter().any(|x| CStr::from_ptr(*x) == e.as_c_str()) {
                    exts.push(e.as_ptr());
                }
            }
            if validation {
                exts.push(ash::ext::debug_utils::NAME.as_ptr());
                layers.push(c"VK_LAYER_KHRONOS_validation".as_ptr());
            }
            let app = vk::ApplicationInfo::default()
                .application_name(c"SmartImposter")
                .engine_name(c"SmartImposter")
                .api_version(vk::API_VERSION_1_3);
            let instance = entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app)
                    .enabled_extension_names(&exts)
                    .enabled_layer_names(&layers),
                None,
            )?;

            let debug = if validation {
                let du = ash::ext::debug_utils::Instance::new(&entry, &instance);
                let m = du.create_debug_utils_messenger(
                    &vk::DebugUtilsMessengerCreateInfoEXT::default()
                        .message_severity(
                            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                        )
                        .message_type(
                            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                        )
                        .pfn_user_callback(Some(debug_cb)),
                    None,
                )?;
                Some((du, m))
            } else {
                None
            };

            let surface = ash_window::create_surface(&entry, &instance, display, window, None)?;
            let surface_fn = ash::khr::surface::Instance::new(&entry, &instance);

            let dev_exts = [
                ash::khr::swapchain::NAME,
                ash::khr::acceleration_structure::NAME,
                ash::khr::ray_tracing_pipeline::NAME,
                ash::khr::deferred_host_operations::NAME,
            ];

            // Pick the best device that supports hardware ray tracing and can present.
            let mut best: Option<(i32, vk::PhysicalDevice, u32)> = None;
            for pd in instance.enumerate_physical_devices()? {
                let props = instance.get_physical_device_properties(pd);
                let avail = instance.enumerate_device_extension_properties(pd)?;
                let has_all = dev_exts.iter().all(|want| {
                    avail.iter().any(|e| e.extension_name_as_c_str().map(|n| n == *want).unwrap_or(false))
                });
                if !has_all {
                    continue;
                }
                let qf = instance
                    .get_physical_device_queue_family_properties(pd)
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                            && surface_fn.get_physical_device_surface_support(pd, i as u32, surface).unwrap_or(false)
                    });
                let Some(qf) = qf else { continue };
                let score = if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU { 2 } else { 1 };
                if best.map(|b| score > b.0).unwrap_or(true) {
                    best = Some((score, pd, qf as u32));
                }
            }
            let (_, pdev, queue_family) = best.ok_or_else(|| anyhow!("no GPU with hardware ray tracing found"))?;
            let props = instance.get_physical_device_properties(pdev);
            let device_name = props.device_name_as_c_str().map(|c| c.to_string_lossy().into_owned()).unwrap_or_default();

            let mut rt_props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
            let mut as_props = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
            {
                let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut rt_props).push_next(&mut as_props);
                instance.get_physical_device_properties2(pdev, &mut p2);
            }
            let rt = RtProps {
                handle_size: rt_props.shader_group_handle_size,
                handle_alignment: rt_props.shader_group_handle_alignment,
                base_alignment: rt_props.shader_group_base_alignment,
                scratch_alignment: as_props.min_acceleration_structure_scratch_offset_alignment as u64,
            };

            let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
                .buffer_device_address(true)
                .descriptor_indexing(true)
                .shader_sampled_image_array_non_uniform_indexing(true)
                .runtime_descriptor_array(true)
                .scalar_block_layout(true);
            let mut f13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true).maintenance4(true);
            let mut as_f = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default().acceleration_structure(true);
            let mut rt_f = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);
            let mut f2 = vk::PhysicalDeviceFeatures2::default()
                .features(vk::PhysicalDeviceFeatures::default().shader_int64(true))
                .push_next(&mut f12)
                .push_next(&mut f13)
                .push_next(&mut as_f)
                .push_next(&mut rt_f);
            let prio = [1.0f32];
            let qci = [vk::DeviceQueueCreateInfo::default().queue_family_index(queue_family).queue_priorities(&prio)];
            let mut ext_ptrs: Vec<*const c_char> = dev_exts.iter().map(|e| e.as_ptr()).collect();
            let avail_dev = instance.enumerate_device_extension_properties(pdev)?;
            for e in &ngx_dev {
                // Core 1.2 bufferDeviceAddress is enabled; the EXT version must not be enabled with it.
                if e.as_c_str() == c"VK_EXT_buffer_device_address" {
                    continue;
                }
                let have = avail_dev.iter().any(|a| a.extension_name_as_c_str().map(|n| n == e.as_c_str()).unwrap_or(false));
                if have && !ext_ptrs.iter().any(|x| CStr::from_ptr(*x) == e.as_c_str()) {
                    ext_ptrs.push(e.as_ptr());
                }
            }
            if !ngx_dev.is_empty() {
                println!("NGX extensions: {}", ngx_dev.iter().map(|e| e.to_string_lossy()).collect::<Vec<_>>().join(" "));
            }
            let device = instance.create_device(
                pdev,
                &vk::DeviceCreateInfo::default().queue_create_infos(&qci).enabled_extension_names(&ext_ptrs).push_next(&mut f2),
                None,
            )?;
            let queue = device.get_device_queue(queue_family, 0);

            let allocator = Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device: pdev,
                debug_settings: Default::default(),
                buffer_device_address: true,
                allocation_sizes: Default::default(),
            })?;
            let cmd_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;

            Ok(Self {
                swapchain_fn: ash::khr::swapchain::Device::new(&instance, &device),
                accel_fn: ash::khr::acceleration_structure::Device::new(&instance, &device),
                rt_fn: ash::khr::ray_tracing_pipeline::Device::new(&instance, &device),
                entry,
                instance,
                debug,
                surface_fn,
                surface,
                pdev,
                device,
                queue,
                queue_family,
                rt,
                allocator: ManuallyDrop::new(Mutex::new(allocator)),
                cmd_pool,
                device_name,
                timestamp_period: props.limits.timestamp_period,
            })
        }
    }

    // ------------------------------------------------------------------ buffers
    pub fn create_buffer(&self, size: u64, usage: vk::BufferUsageFlags, loc: MemoryLocation, name: &str) -> Result<Buffer> {
        let size = size.max(16);
        unsafe {
            let raw = self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;
            let req = self.device.get_buffer_memory_requirements(raw);
            let alloc = self.allocator.lock().unwrap().allocate(&AllocationCreateDesc {
                name,
                requirements: req,
                location: loc,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;
            self.device.bind_buffer_memory(raw, alloc.memory(), alloc.offset())?;
            let addr = self.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(raw));
            Ok(Buffer { raw, alloc: Some(alloc), size, addr })
        }
    }

    pub fn destroy_buffer(&self, b: &mut Buffer) {
        unsafe { self.device.destroy_buffer(b.raw, None) };
        if let Some(a) = b.alloc.take() {
            let _ = self.allocator.lock().unwrap().free(a);
        }
        b.raw = vk::Buffer::null();
    }

    /// Upload `data` into a new device-local buffer through a staging buffer.
    pub fn upload_buffer(&self, data: &[u8], usage: vk::BufferUsageFlags, name: &str) -> Result<Buffer> {
        let mut staging = self.create_buffer(data.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "staging")?;
        staging.mapped()[..data.len()].copy_from_slice(data);
        let dst = self.create_buffer(data.len() as u64, usage | vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuOnly, name)?;
        self.one_time(|cb| unsafe {
            self.device.cmd_copy_buffer(cb, staging.raw, dst.raw, &[vk::BufferCopy::default().size(data.len() as u64)]);
        })?;
        self.destroy_buffer(&mut staging);
        Ok(dst)
    }

    // ------------------------------------------------------------------ images
    pub fn create_image(&self, w: u32, h: u32, mips: u32, format: vk::Format, usage: vk::ImageUsageFlags, name: &str) -> Result<Image> {
        unsafe {
            let raw = self.device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D { width: w, height: h, depth: 1 })
                    .mip_levels(mips)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )?;
            let req = self.device.get_image_memory_requirements(raw);
            let alloc = self.allocator.lock().unwrap().allocate(&AllocationCreateDesc {
                name,
                requirements: req,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;
            self.device.bind_image_memory(raw, alloc.memory(), alloc.offset())?;
            let make_view = |base: u32, count: u32| {
                self.device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(raw)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format)
                        .subresource_range(color_range(base, count)),
                    None,
                )
            };
            let view = make_view(0, mips)?;
            let mut mip_views = Vec::new();
            for m in 0..mips {
                mip_views.push(make_view(m, 1)?);
            }
            Ok(Image { raw, alloc: Some(alloc), view, mip_views, format, width: w, height: h, mips })
        }
    }

    pub fn destroy_image(&self, img: &mut Image) {
        unsafe {
            for v in img.mip_views.drain(..) {
                self.device.destroy_image_view(v, None);
            }
            self.device.destroy_image_view(img.view, None);
            self.device.destroy_image(img.raw, None);
        }
        if let Some(a) = img.alloc.take() {
            let _ = self.allocator.lock().unwrap().free(a);
        }
    }

    // ------------------------------------------------------------------ commands & barriers
    pub fn one_time<R>(&self, f: impl FnOnce(vk::CommandBuffer) -> R) -> Result<R> {
        unsafe {
            let cb = self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            self.device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            let r = f(cb);
            self.device.end_command_buffer(cb)?;
            let fence = self.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cbs = [cb];
            self.device.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cbs)], fence)?;
            let res = self.device.wait_for_fences(&[fence], true, u64::MAX);
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.cmd_pool, &cbs);
            res.context("GPU work failed (device lost?)")?;
            Ok(r)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn image_barrier(
        &self,
        cb: vk::CommandBuffer,
        image: vk::Image,
        old: vk::ImageLayout,
        new: vk::ImageLayout,
        src_stage: vk::PipelineStageFlags2,
        src_access: vk::AccessFlags2,
        dst_stage: vk::PipelineStageFlags2,
        dst_access: vk::AccessFlags2,
    ) {
        let b = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)
            .old_layout(old)
            .new_layout(new)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(color_range(0, vk::REMAINING_MIP_LEVELS))];
        unsafe { self.device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&b)) };
    }

    pub fn memory_barrier(
        &self,
        cb: vk::CommandBuffer,
        src_stage: vk::PipelineStageFlags2,
        src_access: vk::AccessFlags2,
        dst_stage: vk::PipelineStageFlags2,
        dst_access: vk::AccessFlags2,
    ) {
        let b = [vk::MemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)];
        unsafe { self.device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&b)) };
    }

    pub fn shader_module(&self, spv: &[u8]) -> Result<vk::ShaderModule> {
        let code = ash::util::read_spv(&mut std::io::Cursor::new(spv))?;
        Ok(unsafe { self.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)? })
    }

    // ------------------------------------------------------------------ swapchain
    pub fn create_swapchain(&self, width: u32, height: u32, old: vk::SwapchainKHR, vsync: bool) -> Result<Swapchain> {
        unsafe {
            let caps = self.surface_fn.get_physical_device_surface_capabilities(self.pdev, self.surface)?;
            let formats = self.surface_fn.get_physical_device_surface_formats(self.pdev, self.surface)?;
            let fmt = formats
                .iter()
                .find(|f| {
                    matches!(f.format, vk::Format::B8G8R8A8_UNORM | vk::Format::R8G8B8A8_UNORM)
                        && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
                .copied()
                .ok_or_else(|| anyhow!("no UNORM swapchain format"))?;
            let modes = self.surface_fn.get_physical_device_surface_present_modes(self.pdev, self.surface)?;
            let mode = if vsync {
                vk::PresentModeKHR::FIFO
            } else if modes.contains(&vk::PresentModeKHR::MAILBOX) {
                vk::PresentModeKHR::MAILBOX
            } else if modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
                vk::PresentModeKHR::IMMEDIATE
            } else {
                vk::PresentModeKHR::FIFO
            };
            let extent = if caps.current_extent.width != u32::MAX {
                caps.current_extent
            } else {
                vk::Extent2D {
                    width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                    height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
                }
            };
            if extent.width == 0 || extent.height == 0 {
                bail!("zero sized surface");
            }
            let mut count = caps.min_image_count + 1;
            if caps.max_image_count > 0 {
                count = count.min(caps.max_image_count);
            }
            let raw = self.swapchain_fn.create_swapchain(
                &vk::SwapchainCreateInfoKHR::default()
                    .surface(self.surface)
                    .min_image_count(count)
                    .image_format(fmt.format)
                    .image_color_space(fmt.color_space)
                    .image_extent(extent)
                    .image_array_layers(1)
                    .image_usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT)
                    .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .pre_transform(caps.current_transform)
                    .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                    .present_mode(mode)
                    .clipped(true)
                    .old_swapchain(old),
                None,
            )?;
            let images = self.swapchain_fn.get_swapchain_images(raw)?;
            let mut done = Vec::new();
            for _ in &images {
                done.push(self.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?);
            }
            Ok(Swapchain { raw, images, format: fmt.format, extent, done })
        }
    }

    pub fn destroy_swapchain(&self, s: &mut Swapchain) {
        unsafe {
            for sem in s.done.drain(..) {
                self.device.destroy_semaphore(sem, None);
            }
            self.swapchain_fn.destroy_swapchain(s.raw, None);
        }
    }
}

pub fn color_range(base: u32, count: u32) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: base,
        level_count: count,
        base_array_layer: 0,
        layer_count: 1,
    }
}
