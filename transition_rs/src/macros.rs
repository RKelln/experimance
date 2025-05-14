//src/macros.rs

#[macro_export]
macro_rules! time_it {
    ($name:expr, $block:block) => {{
        #[cfg(feature = "timing")]
        {
            use std::time::Instant;
            let start = Instant::now();
            let result = { $block };
            let duration = start.elapsed();
            info!("{} took {:?}", $name, duration);
            result
        }
        #[cfg(not(feature = "timing"))]
        {
            { $block }
        }
    }};
}