with open("turboquant-explained/android/TurboQuantBench/native/src/lib.rs", "r") as f:
    content = f.read()

# Fix IdMapIndex3232
content = content.replace("IdMapIndex3232", "IdMapIndex32")

# Fix nativeIdMapSearchResult long array
content = content.replace("env.set_long_array_region(&j_ids, 0, &ids_i64)", "env.set_int_array_region(&j_ids, 0, &ids_i32)")

# Write back
with open("turboquant-explained/android/TurboQuantBench/native/src/lib.rs", "w") as f:
    f.write(content)
