import re

with open("turboquant-explained/android/TurboQuantBench/native/src/lib.rs", "r") as f:
    content = f.read()

# Add JIntArray
content = content.replace(
    "use jni::objects::{JFloatArray, JLongArray, JObject, JValue};",
    "use jni::objects::{JFloatArray, JIntArray, JLongArray, JObject, JValue};"
)

# Add read_int_array
read_int_array = """
fn read_int_array(env: &mut JNIEnv<'_>, array: &JIntArray<'_>) -> Result<Vec<u32>, String> {
    let len = env.get_array_length(array).map_err(|e| e.to_string())? as usize;
    let mut values = vec![0i32; len];
    env.get_int_array_region(array, 0, &mut values)
        .map_err(|e| e.to_string())?;
    Ok(values.into_iter().map(|v| v as u32).collect())
}

unsafe fn id_map32_ref<'a>(handle: jlong) -> Result<&'a turbovec::IdMapIndex32, String> {
    if handle == 0 {
        return Err("native turbovec id-map32 handle is null".to_string());
    }
    Ok(&*(handle as *const turbovec::IdMapIndex32))
}

unsafe fn id_map32_mut<'a>(handle: jlong) -> Result<&'a mut turbovec::IdMapIndex32, String> {
    if handle == 0 {
        return Err("native turbovec id-map32 handle is null".to_string());
    }
    Ok(&mut *(handle as *mut turbovec::IdMapIndex32))
}
"""
content = content.replace(
    "fn read_float_array",
    read_int_array + "\nfn read_float_array"
)

# Append new 32-bit endpoints
appendix = """
// ─── IdMapIndex32 JNI ────────────────────────────────────────────────────────

#[no_mangle]
pub extern "system" fn Java_com_turbovec_lib_TurboVecIndex_nativeIdMap32New(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    dim: jint,
    bit_width: jint,
) -> jlong {
    if dim <= 0 {
        throw_ex(&mut env, "java/lang/IllegalArgumentException", "dim must be positive");
        return 0;
    }
    if bit_width <= 0 {
        throw_ex(&mut env, "java/lang/IllegalArgumentException", "bit_width must be positive");
        return 0;
    }
    match turbovec::IdMapIndex32::new(dim as usize, bit_width as usize) {
        Ok(index) => Box::into_raw(Box::new(index)) as jlong,
        Err(e) => {
            throw_ex(&mut env, "java/lang/IllegalArgumentException", &format!("{e:?}"));
            0
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_com_turbovec_lib_TurboVecIndex_nativeIdMap32Load(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    path: JString<'_>,
) -> jlong {
    let path_str = match read_string_val(&mut env, &path) {
        Ok(s) => s,
        Err(e) => {
            throw_ex(&mut env, "java/io/IOException", &e);
            return 0;
        }
    };
    match turbovec::IdMapIndex32::load(&path_str) {
        Ok(index) => Box::into_raw(Box::new(index)) as jlong,
        Err(e) => {
            throw_ex(&mut env, "java/io/IOException", &format!("{e:?}"));
            0
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_com_turbovec_lib_TurboVecIndex_nativeIdMap32Free(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
) {
    if handle != 0 {
        unsafe {
            drop(Box::from_raw(handle as *mut turbovec::IdMapIndex32));
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_com_turbovec_lib_TurboVecIndex_nativeIdMap32Add(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
    vectors: JFloatArray<'_>,
    dim: jint,
    ids: JIntArray<'_>,
) {
    let index = match unsafe { id_map32_mut(handle) } {
        Ok(idx) => idx,
        Err(e) => {
            throw_ex(&mut env, "java/lang/IllegalStateException", &e);
            return;
        }
    };
    let vectors_vec = match read_float_array(&mut env, &vectors) {
        Ok(v) => v,
        Err(e) => {
            throw_ex(&mut env, "java/lang/IllegalArgumentException", &e);
            return;
        }
    };
    let ids_vec = match read_int_array(&mut env, &ids) {
        Ok(v) => v,
        Err(e) => {
            throw_ex(&mut env, "java/lang/IllegalArgumentException", &e);
            return;
        }
    };
    if let Err(e) = index.add_with_ids_2d(&vectors_vec, dim as usize, &ids_vec) {
        throw_ex(&mut env, "java/lang/IllegalArgumentException", &format!("{e:?}"));
    }
}

#[no_mangle]
pub extern "system" fn Java_com_turbovec_lib_TurboVecIndex_nativeIdMap32Search(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
    queries: JFloatArray<'_>,
    k: jint,
    out_scores: JFloatArray<'_>,
    out_ids: JIntArray<'_>,
) -> jint {
    let index = match unsafe { id_map32_ref(handle) } {
        Ok(idx) => idx,
        Err(e) => {
            throw_ex(&mut env, "java/lang/IllegalStateException", &e);
            return 0;
        }
    };
    let queries_vec = match read_float_array(&mut env, &queries) {
        Ok(v) => v,
        Err(e) => {
            throw_ex(&mut env, "java/lang/IllegalArgumentException", &e);
            return 0;
        }
    };
    if k < 0 {
        throw_ex(&mut env, "java/lang/IllegalArgumentException", "k must be non-negative");
        return 0;
    }
    let (scores, ids) = index.search(&queries_vec, k as usize);
    let count = scores.len() as jint;
    if count > 0 {
        if let Err(e) = env.set_float_array_region(&out_scores, 0, &scores) {
            throw_ex(&mut env, "java/lang/RuntimeException", &e.to_string());
            return 0;
        }
        let ids_i32: Vec<i32> = ids.into_iter().map(|id| id as i32).collect();
        if let Err(e) = env.set_int_array_region(&out_ids, 0, &ids_i32) {
            throw_ex(&mut env, "java/lang/RuntimeException", &e.to_string());
            return 0;
        }
    }
    count
}
"""
with open("turboquant-explained/android/TurboQuantBench/native/src/lib.rs", "w") as f:
    f.write(content + appendix)

print("Patch applied to android native lib.rs")
