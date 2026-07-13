package com.turbovec.sample;

import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.RadioGroup;
import android.widget.ScrollView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.turbovec.lib.TurboVecIndex;

import java.io.File;
import java.io.IOException;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * TurboVec Explorer — demonstrates all TurboVecIndex (com.turbovec.lib) API operations:
 *
 *  1. Create — new TurboVecIndex(dim, bitWidth, filePath): auto-loads if file exists
 *  2. Load   — explicit TurboVecIndex.load(filePath)
 *  3. Insert — add(float[], dim, int[] ids) with 5 test vectors, IDs 101–105
 *  4. Search — search(queries, k, float[] scores, int[] ids) → returns result count
 *  5. Delete — remove(id) + removeAll(int[])
 *  6. Save   — save() to the default file path
 */
public final class MainActivity extends AppCompatActivity {

    // ─── Default index config ────────────────────────────────────────────────────
    private static final int    DEFAULT_DIM      = 384;
    private static final int    DEFAULT_BIT_WIDTH = 4;
    private static final int    SEARCH_K          = 3;

    // Test vector IDs inserted by "Insert Vectors"
    private static final int[] TEST_IDS = { 101, 102, 103, 104, 105 };

    // ─── State ───────────────────────────────────────────────────────────────────
    private TurboVecIndex index;
    private final ExecutorService executor  = Executors.newSingleThreadExecutor();
    private final Handler         mainHandler = new Handler(Looper.getMainLooper());

    // ─── Views ───────────────────────────────────────────────────────────────────
    private EditText      etDim, etFilePath;
    private RadioGroup    rgBitWidth;
    private Button        btnCreate, btnLoad, btnInsert, btnSearch, btnDelete, btnSave, btnClear;
    private LinearLayout  resultsCard, resultsContainer;
    private TextView      tvResultCount, tvConsole;
    private ScrollView    consoleScroll;

    // ─── Lifecycle ───────────────────────────────────────────────────────────────

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        bindViews();
        wireButtons();

        // Set a sensible default file path
        File defaultFile = new File(getFilesDir(), "turbovec_index.tvim");
        etFilePath.setText(defaultFile.getAbsolutePath());

        log("[System] TurboVec Explorer ready.");
        log("[System] Configure and tap 'Create Index' to begin.");
    }

    @Override
    protected void onDestroy() {
        executor.shutdownNow();
        closeIndex();
        super.onDestroy();
    }

    // ─── View binding ────────────────────────────────────────────────────────────

    private void bindViews() {
        etDim           = findViewById(R.id.et_dim);
        etFilePath      = findViewById(R.id.et_filepath);
        rgBitWidth      = findViewById(R.id.rg_bitwidth);
        btnCreate       = findViewById(R.id.btn_create);
        btnLoad         = findViewById(R.id.btn_load);
        btnInsert       = findViewById(R.id.btn_insert);
        btnSearch       = findViewById(R.id.btn_search);
        btnDelete       = findViewById(R.id.btn_delete);
        btnSave         = findViewById(R.id.btn_save);
        btnClear        = findViewById(R.id.btn_clear);
        resultsCard     = findViewById(R.id.results_card);
        resultsContainer = findViewById(R.id.results_container);
        tvResultCount   = findViewById(R.id.tv_result_count);
        tvConsole       = findViewById(R.id.tv_console);
        consoleScroll   = findViewById(R.id.console_scroll);
    }

    private void wireButtons() {
        btnCreate.setOnClickListener(v -> handleCreate());
        btnLoad  .setOnClickListener(v -> handleLoad());
        btnInsert.setOnClickListener(v -> handleInsert());
        btnSearch.setOnClickListener(v -> handleSearch());
        btnDelete.setOnClickListener(v -> handleDelete());
        btnSave  .setOnClickListener(v -> handleSave());
        btnClear .setOnClickListener(v -> clearConsole());
    }

    // ─── Button handlers ─────────────────────────────────────────────────────────

    /**
     * Create (or auto-load) an index.
     * Uses the constructor: new TurboVecIndex(dim, bitWidth, filePath)
     * → creates fresh if file absent, loads existing if file present.
     */
    private void handleCreate() {
        int dim      = parseDim();
        int bitWidth = parseBitWidth();
        String path  = etFilePath.getText().toString().trim();
        if (dim <= 0 || bitWidth <= 0) return;
        if (TextUtils.isEmpty(path)) { logError("File path is required"); return; }

        setButtonsEnabled(false);
        log("\n[Create] dim=" + dim + "  bitWidth=" + bitWidth);
        log("[Create] File: " + path);

        executor.execute(() -> {
            try {
                closeIndex();
                boolean existed = new File(path).exists();
                index = new TurboVecIndex(dim, bitWidth, path);
                logSafe("[Create] " + (existed ? "Loaded existing" : "Created new")
                        + " index · size=" + index.size()
                        + "  dim=" + index.dim()
                        + "  bitWidth=" + index.bitWidth());
            } catch (Exception e) {
                logError("Create failed: " + e.getMessage());
            } finally {
                setButtonsEnabledSafe(true);
            }
        });
    }

    /**
     * Explicit load via TurboVecIndex.load(filePath).
     */
    private void handleLoad() {
        String path = etFilePath.getText().toString().trim();
        if (TextUtils.isEmpty(path)) { logError("File path is required"); return; }

        setButtonsEnabled(false);
        log("\n[Load] Loading from: " + path);

        executor.execute(() -> {
            try {
                closeIndex();
                index = TurboVecIndex.load(path);
                logSafe("[Load] Loaded · size=" + index.size()
                        + "  dim=" + index.dim()
                        + "  bitWidth=" + index.bitWidth());
            } catch (IOException e) {
                logError("Load failed: " + e.getMessage());
            } finally {
                setButtonsEnabledSafe(true);
            }
        });
    }

    /**
     * Insert 5 synthetic test vectors with IDs 101–105.
     * Each vector is a unit vector pattern in the embedding space.
     */
    private void handleInsert() {
        if (!checkIndexOpen()) return;
        setButtonsEnabled(false);
        log("\n[Insert] Adding 5 test vectors · IDs: 101, 102, 103, 104, 105");

        executor.execute(() -> {
            try {
                int dim = index.dim();
                if (dim <= 0) { logError("Index dim not set. Create or Load first."); return; }

                Random rng = new Random(42);
                // Build 5 distinct synthetic embeddings
                float[] vectors = new float[5 * dim];
                for (int v = 0; v < 5; v++) {
                    float base = (v + 1) * 0.1f;
                    for (int c = 0; c < dim; c++) {
                        vectors[v * dim + c] = base + rng.nextFloat() * 0.05f - 0.025f;
                    }
                    // Normalize each vector
                    double norm = 0;
                    for (int c = 0; c < dim; c++) norm += vectors[v*dim+c] * vectors[v*dim+c];
                    norm = Math.sqrt(norm);
                    if (norm > 0) for (int c = 0; c < dim; c++) vectors[v*dim+c] /= norm;
                }

                long t0 = System.currentTimeMillis();
                index.add(vectors, dim, TEST_IDS);
                // prepare() commits the rotation matrix, centroids, and
                // SIMD-blocked layout; without it search() returns zeros.
                index.prepare();
                long ms = System.currentTimeMillis() - t0;

                logSafe("[Insert] ✓ Inserted 5 vectors in " + ms + " ms  · index size now: " + index.size());
                for (int id : TEST_IDS) {
                    logSafe("[Insert]   ID " + id + " present: " + index.contains(id));
                }
            } catch (Exception e) {
                logError("Insert failed: " + e.getMessage());
            } finally {
                setButtonsEnabledSafe(true);
            }
        });
    }

    /**
     * Search top-k using output-parameter API:
     *   int count = index.search(queries, k, float[] scores, int[] ids)
     */
    private void handleSearch() {
        if (!checkIndexOpen()) return;
        setButtonsEnabled(false);
        log("\n[Search] Searching top-" + SEARCH_K + " nearest vectors...");

        executor.execute(() -> {
            try {
                int dim = index.dim();
                if (dim <= 0) { logError("Index dim not set. Insert vectors first."); return; }
                if (index.size() == 0) { logError("Index is empty. Insert vectors first."); return; }

                // Build a query near vector 0.2 (closest to ID=102)
                Random rng = new Random(99);
                float[] query = new float[dim];
                float base = 0.2f;
                double norm = 0;
                for (int c = 0; c < dim; c++) {
                    query[c] = base + rng.nextFloat() * 0.05f - 0.025f;
                    norm += query[c] * query[c];
                }
                norm = Math.sqrt(norm);
                if (norm > 0) for (int c = 0; c < dim; c++) query[c] /= norm;

                // Pre-allocate output buffers  (1 query × k)
                float[] outScores = new float[SEARCH_K];
                long[]  outIds    = new long[SEARCH_K];

                long t0 = System.nanoTime();
                int count = index.search(query, SEARCH_K, outScores, outIds);
                double ms = (System.nanoTime() - t0) / 1_000_000.0;

                logSafe(String.format(Locale.US,
                        "[Search] ✓ %d result(s) in %.3f ms", count, ms));

                // Show results in the card
                final int finalCount = count;
                final float[]  sc = outScores.clone();
                final long[]   ids = outIds.clone();
                mainHandler.post(() -> showResultsCard(finalCount, sc, ids));

            } catch (Exception e) {
                logError("Search failed: " + e.getMessage());
            } finally {
                setButtonsEnabledSafe(true);
            }
        });
    }

    /**
     * Delete ID 103 (single) then removeAll([104, 999]).
     * Demonstrates both remove() and removeAll(); 999 is deliberately absent.
     */
    private void handleDelete() {
        if (!checkIndexOpen()) return;
        setButtonsEnabled(false);
        log("\n[Delete] Removing ID 103 (single remove)...");
        log("[Delete] Removing IDs [104, 999] (batch removeAll)...");

        executor.execute(() -> {
            try {
                boolean removed103 = index.remove(103L);
                logSafe("[Delete] remove(103) → " + removed103 + "  · index size: " + index.size());

                int removedBatch = index.removeAll(new int[]{104, 999});
                logSafe("[Delete] removeAll([104, 999]) → removed " + removedBatch + "  · index size: " + index.size());

                // Verify
                logSafe("[Delete] contains(103): " + index.contains(103L) + "  (expected false)");
                logSafe("[Delete] contains(104): " + index.contains(104L) + "  (expected false)");
                logSafe("[Delete] contains(999): " + index.contains(999L) + "  (expected false)");
                logSafe("[Delete] contains(101): " + index.contains(101L) + "  (expected true)");
                logSafe("[Delete] contains(102): " + index.contains(102L) + "  (expected true)");
                logSafe("[Delete] contains(105): " + index.contains(105L) + "  (expected true)");
            } catch (Exception e) {
                logError("Delete failed: " + e.getMessage());
            } finally {
                setButtonsEnabledSafe(true);
            }
        });
    }

    /**
     * Save via index.save() — uses the default file path set at construction.
     */
    private void handleSave() {
        if (!checkIndexOpen()) return;
        setButtonsEnabled(false);
        String path = etFilePath.getText().toString().trim();
        log("\n[Save] Saving index to: " + path);

        executor.execute(() -> {
            try {
                if (TextUtils.isEmpty(path)) {
                    index.save();  // uses constructor path
                } else {
                    index.save(path);
                }
                long fileSize = new File(path).length();
                logSafe(String.format(Locale.US,
                        "[Save] ✓ Saved  size=%d vectors  file=%.1f KB",
                        index.size(), fileSize / 1024.0));
            } catch (IOException e) {
                logError("Save failed: " + e.getMessage());
            } finally {
                setButtonsEnabledSafe(true);
            }
        });
    }

    // ─── UI helpers ──────────────────────────────────────────────────────────────

    private void showResultsCard(int count, float[] scores, int[] ids) {
        resultsCard.setVisibility(View.VISIBLE);
        resultsContainer.removeAllViews();
        tvResultCount.setText(count + " result" + (count != 1 ? "s" : ""));

        LayoutInflater inflater = LayoutInflater.from(this);
        for (int i = 0; i < count; i++) {
            View item = inflater.inflate(R.layout.item_result, resultsContainer, false);

            TextView tvRank  = item.findViewById(R.id.tv_rank);
            TextView tvId    = item.findViewById(R.id.tv_id);
            TextView tvScore = item.findViewById(R.id.tv_score);

            tvRank.setText(String.valueOf(i + 1));
            tvId.setText("ID: " + ids[i]);
            tvScore.setText(String.format(Locale.US, "Score: %.6f", scores[i]));

            resultsContainer.addView(item);
            log(String.format(Locale.US, "[Search]   #%d  id=%-6d  score=%.6f", i + 1, ids[i], scores[i]));
        }
    }

    private void clearConsole() {
        tvConsole.setText("[System] Console cleared.\n");
    }

    private void log(final String message) {
        mainHandler.post(() -> {
            tvConsole.append(message + "\n");
            consoleScroll.post(() -> consoleScroll.fullScroll(View.FOCUS_DOWN));
        });
    }

    private void logSafe(String message) {
        mainHandler.post(() -> {
            tvConsole.append(message + "\n");
            consoleScroll.post(() -> consoleScroll.fullScroll(View.FOCUS_DOWN));
        });
    }

    private void logError(String message) {
        mainHandler.post(() -> {
            tvConsole.append("[ERROR] " + message + "\n");
            consoleScroll.post(() -> consoleScroll.fullScroll(View.FOCUS_DOWN));
        });
    }

    private void setButtonsEnabled(boolean enabled) {
        btnCreate.setEnabled(enabled);
        btnLoad.setEnabled(enabled);
        btnInsert.setEnabled(enabled);
        btnSearch.setEnabled(enabled);
        btnDelete.setEnabled(enabled);
        btnSave.setEnabled(enabled);
    }

    private void setButtonsEnabledSafe(boolean enabled) {
        mainHandler.post(() -> setButtonsEnabled(enabled));
    }

    private boolean checkIndexOpen() {
        if (index == null || index.isClosed()) {
            logError("No index open. Create or Load an index first.");
            return false;
        }
        return true;
    }

    private void closeIndex() {
        if (index != null && !index.isClosed()) index.close();
        index = null;
    }

    // ─── Config parsers ──────────────────────────────────────────────────────────

    private int parseDim() {
        try {
            int d = Integer.parseInt(etDim.getText().toString().trim());
            if (d <= 0 || d % 8 != 0) {
                logError("Dimension must be > 0 and a multiple of 8"); return -1;
            }
            return d;
        } catch (NumberFormatException e) {
            logError("Invalid dimension value"); return -1;
        }
    }

    private int parseBitWidth() {
        int id = rgBitWidth.getCheckedRadioButtonId();
        if (id == R.id.rb_2bit) return 2;
        if (id == R.id.rb_3bit) return 3;
        if (id == R.id.rb_4bit) return 4;
        if (id == R.id.rb_8bit) return 8;
        logError("Select a bit width"); return -1;
    }
}
