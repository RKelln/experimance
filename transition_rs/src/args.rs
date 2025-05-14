use clap::Parser;

/// Image Processing CLI Application
#[derive(Parser, Debug)]
#[command(author, version, about = "A CLI tool for creating transitions between two images", long_about = None)]
pub struct Args {
    /// Input image path
    #[arg(short = 'i', long, value_name = "FILE", required = true, help = "Path to the input image file")]
    pub input: String,

    /// Output image path
    #[arg(short = 'o', long, value_name = "FILE", default_value = "", help = "Path to the input image file")]
    pub output: String,

    // Size of the processed image
    #[arg(short = 's', long, default_value_t = 0, help = "Size of the processed image (0 = disable)")]
    pub size: i32,

    /// Number of colors for posterization
    #[arg(short = 'c', long, default_value_t = 0, help = "Number of colors to reduce the image to (0 = disable)")]
    pub colors: usize,

    /// Enable LAB color conversion
    #[arg(short = 'L', long, default_value_t = true, help = "Convert image to LAB color space (default: true)")]
    pub lab: bool,

    /// Start location for flowmap generation (format: x,y)
    /// Format: x,y
    /// Example: 0,0
    /// Default: random
    #[arg(short = 'p', long, value_name = "X,Y", help = "Start pixel location for flowmap (format: x,y). Default: random")]
    pub start: Option<String>,
}
