// benches/benchmark_flood_fill_opt.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion, SamplingMode};

use ndarray::Array1;
use image_transition::transitions::flood_fill_opt::{minmax_normalize, minmax_normalize_2, flood_fill_color_image};



fn benchmark_normalize(c: &mut Criterion) {
    // Create a large gradient_map for benchmarking
    let size = 1_000_000; // 1 million elements
    let gradient_map = Array1::from_shape_vec(size, (0..size).map(|x| x as f32).collect()).unwrap();
    
    c.bench_function("minmax_normalize", |b| {
        b.iter(|| {
            let mut map = gradient_map.clone();
            minmax_normalize(&mut map);
            black_box(map);
        })
    });
}

fn benchmark_normalize_2(c: &mut Criterion) {
    let size = 1_000_000;
    let gradient_map = Array1::from_shape_vec(size, (0..size).map(|x| x as f32).collect()).unwrap();
    
    c.bench_function("minmax_normalize_2", |b| {
        b.iter(|| {
            let mut map = gradient_map.clone();
            minmax_normalize_2(&mut map);
            black_box(map);
        })
    });
}

fn benchmark_flood_fill_opt(c: &mut Criterion) {
    use opencv::core::{Mat, MatTrait, Scalar, Vec3b, CV_8UC3};
    use rand::Rng; // Replace with your actual crate/module path

    let mut group = c.benchmark_group("sample-size-example");
    // Configure Criterion.rs to detect smaller differences and increase sample size to improve
    // precision and counteract the resulting noise.
    group.sample_size(10).sampling_mode(SamplingMode::Flat);

    // create 1024 x 1024 Mat with random colors
    let mat = Mat::new_rows_cols_with_default(1024, 1024, CV_8UC3, Scalar::all(0.0)).unwrap();
    // randomize the colors
    let mut rng = rand::thread_rng();
    let mut image = mat.clone();
    for y in 0..1024 {
        for x in 0..1024 {
            let color = Vec3b::from([rng.gen_range(0..255), rng.gen_range(0..255), rng.gen_range(0..255)]);
            *image.at_2d_mut::<Vec3b>(y, x).unwrap() = color;
        }
    }
    let start_points = vec![(2, 2)];

    group.bench_function("flood_fill_color_image", |b| {
        b.iter(|| {
            let flow_map = flood_fill_color_image(&image, &start_points);
            black_box(flow_map.unwrap());
        })
    });
    group.finish();
}

criterion_group!(benches, benchmark_normalize, benchmark_normalize_2, benchmark_flood_fill_opt);
criterion_main!(benches);