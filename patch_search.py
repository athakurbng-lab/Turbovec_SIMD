import re

with open("turboquant-explained/turbovec/turbovec/src/search.rs", "r") as f:
    content = f.read()

# 1. Change search_8bit_scalar return type
content = content.replace(
    ") -> Vec<(Vec<f32>, Vec<i64>)> {",
    ") -> Vec<(Vec<f32>, Vec<u32>)> {"
)
# 2. Inside search_8bit_scalar
content = content.replace(
    "pairs.iter().map(|p| p.1 as i64).collect(),",
    "pairs.iter().map(|p| p.1).collect(),"
)
# 3. Rename search to search_u32 and change return type
content = content.replace(
    "pub fn search(\n    queries: &[f32],",
    "pub fn search_u32(\n    queries: &[f32],"
)
content = content.replace(
    "mask: Option<&[u64]>,\n) -> (Vec<f32>, Vec<i64>) {",
    "mask: Option<&[u64]>,\n) -> (Vec<f32>, Vec<u32>) {"
)
content = content.replace(
    "all_indices.extend(std::iter::repeat(0i64).take(pad));",
    "all_indices.extend(std::iter::repeat(0u32).take(pad));"
)
content = content.replace(
    "let results: Vec<Vec<(Vec<f32>, Vec<i64>)>> = (0..nq)",
    "let results: Vec<Vec<(Vec<f32>, Vec<u32>)>> = (0..nq)"
)
content = content.replace(
    "let i: Vec<i64> = pairs.iter().map(|p| p.1 as i64).collect();",
    "let i: Vec<u32> = pairs.iter().map(|p| p.1).collect();"
)
content = content.replace(
    "let results: Vec<(Vec<f32>, Vec<i64>)> = (0..nq)",
    "let results: Vec<(Vec<f32>, Vec<u32>)> = (0..nq)"
)
content = content.replace(
    "pairs.iter().map(|p| p.1 as i64).collect::<Vec<i64>>(),",
    "pairs.iter().map(|p| p.1).collect::<Vec<u32>>(),"
)

# Append pub fn search wrapper and map_ids_avx2
appendix = """
/// Legacy search wrapper for backward compatibility.
pub fn search(
    queries: &[f32],
    nq: usize,
    rotation: &[f32],
    blocked_codes: &[u8],
    centroids: &[f32],
    vec_scales: &[f32],
    tqplus_shift: &[f32],
    tqplus_scale: &[f32],
    bits: usize,
    dim: usize,
    n_vectors: usize,
    n_blocks: usize,
    k: usize,
    mask: Option<&[u64]>,
) -> (Vec<f32>, Vec<i64>) {
    let (scores, indices_u32) = search_u32(
        queries, nq, rotation, blocked_codes, centroids, vec_scales,
        tqplus_shift, tqplus_scale, bits, dim, n_vectors, n_blocks, k, mask
    );
    let indices_i64 = indices_u32.into_iter().map(|x| x as i64).collect();
    (scores, indices_i64)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn map_ids_avx2(slots: &[u32], slot_to_id: &[u32], out: &mut [u32]) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= slots.len() {
        let idx = _mm256_loadu_si256(slots.as_ptr().add(i) as *const __m256i);
        let gathered = _mm256_i32gather_epi32(slot_to_id.as_ptr() as *const i32, idx, 4);
        _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, gathered);
        i += 8;
    }
    for j in i..slots.len() {
        out[j] = slot_to_id[slots[j] as usize];
    }
}
"""
with open("turboquant-explained/turbovec/turbovec/src/search.rs", "w") as f:
    f.write(content + appendix)

print("Patch applied to search.rs")
