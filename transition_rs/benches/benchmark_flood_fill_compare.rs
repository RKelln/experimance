use criterion::{black_box, criterion_group, criterion_main, Criterion, SamplingMode};
use opencv::core::{Mat, MatTrait, Scalar, Vec3b, CV_8UC3};
use rand::Rng;

use image_transition::transitions::{
    flood_fill_opt,
    flood_fill_opt2,
    flood_fill_opt3,
};

fn create_test_image(size: i32) -> Mat {
    let mut image = Mat::new_rows_cols_with_default(size, size, CV_8UC3, Scalar::all(0.0)).unwrap();
    let mut rng = rand::thread_rng();
    
    for y in 0..size {
        for x in 0..size {
            let color = Vec3b::from([
                rng.gen_range(0..255),
                rng.gen_range(0..255),
                rng.gen_range(0..255)
            ]);
            *image.at_2d_mut::<Vec3b>(y, x).unwrap() = color;
        }
    }
    image
}

fn benchmark_flood_fill_compare(c: &mut Criterion) {
    let mut group = c.benchmark_group("flood_fill_comparison");
    group.sample_size(10)
         .sampling_mode(SamplingMode::Flat);

    let sizes = [256, 512, 1024, 2048, 4096];
    
    for size in sizes {
        let image = create_test_image(size);
        let start_points = vec![(size/2, size/2)];
        
        group.bench_function(format!("opt1_{}x{}", size, size), |b| {
            b.iter(|| {
                let result = flood_fill_opt::flood_fill_color_image(&image, &start_points);
                black_box(result.unwrap());
            })
        });

        group.bench_function(format!("opt2_{}x{}", size, size), |b| {
            b.iter(|| {
                let result = flood_fill_opt2::flood_fill_color_image(&image, &start_points);
                black_box(result.unwrap());
            })
        });

        group.bench_function(format!("opt3_{}x{}", size, size), |b| {
            b.iter(|| {
                let result = flood_fill_opt3::flood_fill_color_image(&image, &start_points);
                black_box(result.unwrap());
            })
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_flood_fill_compare);
criterion_main!(benches);
