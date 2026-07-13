import re

with open("turboquant-explained/turbovec/turbovec/src/io.rs", "r") as f:
    content = f.read()

# Add TV32_MAGIC
content = content.replace(
    "const TV_MAGIC: &[u8; 4] = b\"TVPI\";",
    "const TV_MAGIC: &[u8; 4] = b\"TVPI\";\nconst TV32_MAGIC: &[u8; 4] = b\"TV32\";"
)

appendix = """
pub fn write_id_map32(
    path: impl AsRef<Path>,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    packed_codes: &[u8],
    scales: &[f32],
    tqplus_shift: &[f32],
    tqplus_scale: &[f32],
    slot_to_id: &[u32],
) -> io::Result<()> {
    assert_eq!(
        slot_to_id.len(),
        n_vectors,
        "slot_to_id length {} does not match n_vectors {}",
        slot_to_id.len(),
        n_vectors,
    );

    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(TV32_MAGIC)?;
    f.write_all(&[TVIM_VERSION])?;
    write_core(
        &mut f, bit_width, dim, n_vectors, packed_codes, scales,
        tqplus_shift, tqplus_scale,
    )?;

    for &id in slot_to_id {
        f.write_all(&id.to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}

pub fn load_id_map32(
    path: impl AsRef<Path>,
) -> io::Result<(usize, usize, usize, Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<u32>)> {
    let mut f = BufReader::new(File::open(path)?);

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != TV32_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a TV32 file: wrong magic",
        ));
    }
    let mut version = [0u8; 1];
    f.read_exact(&mut version)?;
    if version[0] == 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "this .tv32 file was written by turbovec ≤ 0.4.3 (format version 1). {}",
                REBUILD_HINT,
            ),
        ));
    }
    let (bit_width, dim, n_vectors, packed_codes, scales, tqplus_shift, tqplus_scale) =
        read_core_versioned(&mut f, version[0], TVIM_VERSION, ".tv32")?;

    let id_bytes = n_vectors
        .checked_mul(4)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "id table size overflows usize"))?;
    let raw = read_exact_vec(&mut f, id_bytes)?;
    let slot_to_id: Vec<u32> = raw
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    Ok((
        bit_width, dim, n_vectors, packed_codes, scales, tqplus_shift, tqplus_scale,
        slot_to_id,
    ))
}
"""
with open("turboquant-explained/turbovec/turbovec/src/io.rs", "w") as f:
    f.write(content + appendix)

print("Patch applied to io.rs")
