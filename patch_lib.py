import re

with open("turboquant-explained/turbovec/turbovec/src/lib.rs", "r") as f:
    content = f.read()

# Add SearchResultsU32
content = content.replace(
    "pub struct SearchResults {",
    "pub struct SearchResultsU32 {\n    pub scores: Vec<f32>,\n    pub indices: Vec<u32>,\n    pub nq: usize,\n    pub k: usize,\n}\n\n#[derive(Debug)]\npub struct SearchResults {"
)

# Replace search_with_mask body to call search::search_u32
old_search = """        let (scores, indices) = search::search(
            queries,
            nq,
            rotation,
            &blocked.data,
            centroids,
            &self.scales,
            &self.tqplus_shift,
            &self.tqplus_scale,
            self.bit_width,
            dim,
            self.n_vectors,
            blocked.n_blocks,
            k,
            packed_mask.as_deref(),
        );

        SearchResults {
            scores,
            indices,
            nq,
            k: effective_k,
        }"""

new_search = """        let (scores, indices) = search::search(
            queries,
            nq,
            rotation,
            &blocked.data,
            centroids,
            &self.scales,
            &self.tqplus_shift,
            &self.tqplus_scale,
            self.bit_width,
            dim,
            self.n_vectors,
            blocked.n_blocks,
            k,
            packed_mask.as_deref(),
        );

        SearchResults {
            scores,
            indices,
            nq,
            k: effective_k,
        }
    }

    pub fn search_u32(&self, queries: &[f32], k: usize) -> SearchResultsU32 {
        self.search_with_mask_u32(queries, k, None)
    }

    pub fn search_with_mask_u32(
        &self,
        queries: &[f32],
        k: usize,
        mask: Option<&[bool]>,
    ) -> SearchResultsU32 {
        let Some(dim) = self.dim else {
            return SearchResultsU32 {
                scores: Vec::new(),
                indices: Vec::new(),
                nq: 0,
                k: 0,
            };
        };
        let nq = queries.len() / dim;
        assert_eq!(queries.len(), nq * dim);
        if let Some((vi, ci, v)) = first_invalid_coord(queries, dim) {
            panic!(
                "invalid query value at query {vi}, coord {ci}: {v} (must be finite and |value| < 1e16 to avoid f32 overflow)",
            );
        }

        let rotation = self
            .rotation
            .get_or_init(|| rotation::make_rotation_matrix(dim));
        let centroids = self.centroids.get_or_init(|| {
            let (_, c) = codebook::codebook(self.bit_width, dim);
            c
        });
        let blocked = self.blocked.get_or_init(|| {
            let (data, n_blocks) =
                pack::repack(&self.packed_codes, self.n_vectors, self.bit_width, dim);
            BlockedCache { data, n_blocks }
        });

        let packed_mask = mask.map(|m| {
            assert_eq!(
                m.len(),
                self.n_vectors,
                "mask length {} does not match index size {}",
                m.len(),
                self.n_vectors,
            );
            let n_words = (self.n_vectors + 63) / 64;
            let mut buf = vec![0u64; n_words];
            for (i, &b) in m.iter().enumerate() {
                if b {
                    buf[i >> 6] |= 1u64 << (i & 63);
                }
            }
            buf
        });

        let n_allowed = packed_mask.as_ref().map_or(self.n_vectors, |p| {
            p.iter().map(|w| w.count_ones() as usize).sum::<usize>()
        });
        let effective_k = k.min(self.n_vectors).min(n_allowed);

        let (scores, indices) = search::search_u32(
            queries,
            nq,
            rotation,
            &blocked.data,
            centroids,
            &self.scales,
            &self.tqplus_shift,
            &self.tqplus_scale,
            self.bit_width,
            dim,
            self.n_vectors,
            blocked.n_blocks,
            k,
            packed_mask.as_deref(),
        );

        SearchResultsU32 {
            scores,
            indices,
            nq,
            k: effective_k,
        }"""

content = content.replace(old_search, new_search)

# Export IdMapIndex32
content = content.replace(
    "pub use id_map::IdMapIndex;",
    "pub use id_map::IdMapIndex;\npub mod id_map32;\npub use id_map32::IdMapIndex32;"
)

with open("turboquant-explained/turbovec/turbovec/src/lib.rs", "w") as f:
    f.write(content)

print("Patch applied to lib.rs")
