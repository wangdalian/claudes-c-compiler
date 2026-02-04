#![recursion_limit = "512"]
// Compiler functions naturally accumulate parameters (context, types, spans, flags).
// Refactoring every one into a struct would add boilerplate without improving clarity.
#![allow(clippy::too_many_arguments)]
// Complex return types arise naturally in compiler data structures; type aliases
// would just move the complexity elsewhere.
#![allow(clippy::type_complexity)]
// Peephole passes use index-based iteration over instruction arrays where the loop
// variable is used as both an index and for bounds arithmetic. Converting to iterators
// would obscure the sliding-window logic.
#![allow(clippy::needless_range_loop)]

pub(crate) mod common;
pub(crate) mod frontend;
pub(crate) mod ir;
pub(crate) mod passes;
pub mod backend;
pub mod driver;

/// Shared entry point for all compiler binaries. Spawns the real work on a
/// thread with a large stack so deeply recursive C files don't overflow.
pub fn compiler_main() {
    const STACK_SIZE: usize = 64 * 1024 * 1024; // 64 MB
    let builder = std::thread::Builder::new().stack_size(STACK_SIZE);
    let handler = builder.spawn(|| {
        let args: Vec<String> = std::env::args().collect();
        let mut driver = driver::Driver::new();
        if driver.parse_cli_args(&args)? {
            return Ok(());
        }
        if !driver.has_input_files() {
            return Err("no input files".to_string());
        }
        driver.run()
    })
    .expect("failed to spawn main thread");

    match handler.join() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            eprintln!("ccc: error: {}", e);
            std::process::exit(1);
        }
        Err(e) => {
            if let Some(s) = e.downcast_ref::<&str>() {
                eprintln!("ccc: internal error: {}", s);
            } else if let Some(s) = e.downcast_ref::<String>() {
                eprintln!("ccc: internal error: {}", s);
            } else {
                eprintln!("ccc: internal error (thread panicked)");
            }
            std::process::exit(1);
        }
    }
}
