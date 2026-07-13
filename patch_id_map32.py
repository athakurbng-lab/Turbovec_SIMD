import re

with open("turboquant-explained/turbovec/turbovec/src/id_map32.rs", "r") as f:
    content = f.read()

# 1. Replace type names and u64 with u32
content = content.replace("IdMapIndex", "IdMapIndex32")
content = content.replace("u64", "u32")

# 2. Replace io::load_id_map and io::write_id_map
content = content.replace("io::load_id_map", "io::load_id_map32")
content = content.replace("io::write_id_map", "io::write_id_map32")

# 3. Modify search_with_allowlist to use search_with_mask_u32 and SIMD map
old_search_block = """        let res = self
            .inner
            .search_with_mask(queries, k, mask_buf.as_deref());

        let mut ids = Vec::with_capacity(res.indices.len());
        for &slot in &res.indices {
            // Inner returns i64 slot indices. Convert via slot_to_id.
            // Slot indices are always in-bounds (the kernel never
            // returns negative or out-of-range values for a valid
            // index), so this lookup cannot fail in practice; the
            // bounds check makes that invariant crash-loud if it ever
            // does.
            let id = self.slot_to_id[slot as usize];
            ids.push(id);
        }
        (res.scores, ids)"""

new_search_block = """        let res = self
            .inner
            .search_with_mask_u32(queries, k, mask_buf.as_deref());

        let mut ids = vec![0u32; res.indices.len()];
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { crate::search::map_ids_avx2(&res.indices, &self.slot_to_id, &mut ids) };
            } else {
                for i in 0..res.indices.len() {
                    ids[i] = self.slot_to_id[res.indices[i] as usize];
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            for i in 0..res.indices.len() {
                ids[i] = self.slot_to_id[res.indices[i] as usize];
            }
        }
        (res.scores, ids)"""

content = content.replace(old_search_block, new_search_block)

with open("turboquant-explained/turbovec/turbovec/src/id_map32.rs", "w") as f:
    f.write(content)

print("Created and patched id_map32.rs")
