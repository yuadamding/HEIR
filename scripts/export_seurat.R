#!/usr/bin/env Rscript

# Export an annotated Seurat v4/v5 object to H5AD without modifying the source.
# Run with the existing environment:
#   conda run -n r_env Rscript scripts/export_seurat.R input.rds output.h5ad RNA

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2 || length(args) > 3) {
  stop("usage: export_seurat.R INPUT.rds OUTPUT.h5ad [ASSAY]")
}

input_path <- normalizePath(args[[1]], mustWork = TRUE)
output_path <- normalizePath(dirname(args[[2]]), mustWork = TRUE)
output_path <- file.path(output_path, basename(args[[2]]))
assay_name <- if (length(args) == 3) args[[3]] else NULL

suppressPackageStartupMessages({
  library(Seurat)
  library(SingleCellExperiment)
  library(S4Vectors)
  library(zellkonverter)
  library(jsonlite)
})

sha256_file <- function(path) {
  output <- system2("sha256sum", args = path, stdout = TRUE, stderr = TRUE)
  if (length(output) != 1 || !grepl("^[0-9a-f]{64}", output[[1]])) {
    stop(sprintf("could not SHA-256 hash %s", path))
  }
  strsplit(output[[1]], "[[:space:]]+")[[1]][[1]]
}

source_sha256 <- sha256_file(input_path)

object <- readRDS(input_path)
if (!inherits(object, "Seurat")) {
  stop("input object is not a Seurat object")
}
if (!is.null(assay_name)) {
  if (!(assay_name %in% names(object@assays))) {
    stop(sprintf("assay '%s' is unavailable; choices: %s", assay_name,
                 paste(names(object@assays), collapse = ", ")))
  }
  DefaultAssay(object) <- assay_name
}

# SeuratDisk still calls the removed SeuratObject `slot=` API in current v5
# environments. Build a minimal SingleCellExperiment from the v5 layer API and
# let zellkonverter write the standards-compliant sparse H5AD directly.
counts <- LayerData(object, assay = DefaultAssay(object), layer = "counts")
if (nrow(counts) == 0 || ncol(counts) == 0) {
  stop("selected assay has no count layer")
}
metadata <- object[[]]
metadata <- metadata[colnames(counts), , drop = FALSE]
sce <- SingleCellExperiment(
  assays = list(counts = counts),
  colData = DataFrame(metadata)
)
rowData(sce)$feature_name <- rownames(counts)
if (file.exists(output_path)) {
  file.remove(output_path)
}
writeH5AD(sce, output_path, X_name = "counts", compression = "gzip")
if (!file.exists(output_path)) {
  stop(sprintf("zellkonverter did not produce %s", output_path))
}

provenance_path <- paste0(output_path, ".provenance.json")
write_json(
  list(
    schema = "heir.seurat_conversion.v1",
    source_path = input_path,
    source_sha256 = source_sha256,
    derivative_path = normalizePath(output_path),
    derivative_sha256 = sha256_file(output_path),
    assay = DefaultAssay(object),
    observations = ncol(object),
    genes = nrow(object)
  ),
  provenance_path,
  auto_unbox = TRUE,
  pretty = TRUE
)

cat(sprintf("wrote %s with %d observations and %d genes; provenance %s\n",
            output_path, ncol(object), nrow(object), provenance_path))
