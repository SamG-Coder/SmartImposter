//! Compiles every ray tracing / compute shader in `shaders/` to SPIR-V with glslangValidator.

use std::{env, path::PathBuf, process::Command};

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let glslang = env::var("VULKAN_SDK")
        .ok()
        .map(|s| PathBuf::from(s).join("Bin").join("glslangValidator.exe"))
        .filter(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from("glslangValidator"));

    println!("cargo:rerun-if-changed=shaders");
    for entry in std::fs::read_dir("shaders").expect("shaders dir") {
        let path = entry.unwrap().path();
        println!("cargo:rerun-if-changed={}", path.display());
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !matches!(ext, "rgen" | "rmiss" | "rchit" | "rint" | "rahit" | "comp") {
            continue;
        }
        let name = path.file_name().unwrap().to_str().unwrap();
        let dst = out.join(format!("{name}.spv"));
        let result = Command::new(&glslang)
            .args(["--target-env", "vulkan1.3", "-V", "-Ishaders"])
            .arg(&path)
            .arg("-o")
            .arg(&dst)
            .output()
            .expect("failed to run glslangValidator (is the Vulkan SDK installed?)");
        if !result.status.success() {
            panic!(
                "shader {name} failed to compile:\n{}{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
        }
    }
}
