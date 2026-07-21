#!/usr/bin/env Rscript

# Export the processed spatialDLPFC Visium/snRNA inputs needed by the explicitly
# exploratory, non-confirmatory biological-hypothesis run.  Only raw UMI counts,
# structural identifiers/coordinates, broad cell labels, and embedded hires H&E
# images are read.  Outcome-derived layers, clusters, deconvolution, and logcounts
# are deliberately not exported.

options(stringsAsFactors = FALSE, warn = 1)

script_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
repo_root <- normalizePath(file.path(dirname(script_file), ".."), mustWork = TRUE)

defaults <- list(
  spe = "/mnt/seagate/HnE/spatialDLPFC/pilot_acquisition/opaque/spe_filtered_final_with_clusters_and_deconvolution_results.rds",
  sce = "/mnt/seagate/HnE/spatialDLPFC/pilot_acquisition/reference_qualification/sce_DLPFC_annotated/se.rds",
  sce_assays = "/mnt/seagate/HnE/spatialDLPFC/pilot_acquisition/reference_qualification/sce_DLPFC_annotated/assays.h5",
  panel = file.path(repo_root, "configs", "spatialdlpfc_exploratory_gene_panel.json"),
  protocol = file.path(repo_root, "configs", "spatialdlpfc_exploratory_protocol_v2.json"),
  output_root = "/mnt/seagate/HEIR_runs/spatialdlpfc_exploratory/source",
  block_size = 8192L,
  preflight = FALSE,
  overwrite = FALSE
)

frozen_protocol_sha256 <- "7a33d4e1b786bf24fd0c17a1cbefffc1ef6365fd241e50ad863dfa23e7dd430e"

usage <- function() {
  cat(paste0(
    "Usage: Rscript scripts/export_spatialdlpfc_exploratory_source.R [options]\n",
    "  --spe PATH          processed SpatialExperiment RDS\n",
    "  --sce PATH          HDF5-backed SingleCellExperiment RDS\n",
    "  --sce-assays PATH   HDF5 assay payload paired with --sce\n",
    "  --panel PATH        protocol-frozen compatibility panel JSON\n",
    "  --protocol PATH     frozen exploratory protocol JSON\n",
    "  --output-root PATH  output directory (source.h5, images/, receipt.json)\n",
    "  --block-size N      observation block size (default 8192)\n",
    "  --preflight         validate identities/schema/metadata without writing outputs\n",
    "  --overwrite         replace completed outputs in this output root\n"
  ))
}

parse_args <- function(args) {
  result <- defaults
  value_options <- c(
    "--spe" = "spe", "--sce" = "sce", "--sce-assays" = "sce_assays",
    "--panel" = "panel", "--protocol" = "protocol",
    "--output-root" = "output_root", "--block-size" = "block_size"
  )
  i <- 1L
  while (i <= length(args)) {
    token <- args[[i]]
    if (token %in% c("-h", "--help")) {
      usage()
      quit(status = 0L)
    }
    if (identical(token, "--overwrite")) {
      result$overwrite <- TRUE
      i <- i + 1L
      next
    }
    if (identical(token, "--preflight")) {
      result$preflight <- TRUE
      i <- i + 1L
      next
    }
    if (!(token %in% names(value_options)) || i == length(args)) {
      stop("unknown option or missing value: ", token, call. = FALSE)
    }
    result[[value_options[[token]]]] <- args[[i + 1L]]
    i <- i + 2L
  }
  result$block_size <- suppressWarnings(as.integer(result$block_size))
  if (is.na(result$block_size) || result$block_size < 256L) {
    stop("--block-size must be an integer >= 256", call. = FALSE)
  }
  result
}

args <- parse_args(commandArgs(TRUE))

required_packages <- c(
  "DelayedArray", "HDF5Array", "jsonlite", "magick", "rhdf5", "SpatialExperiment",
  "SummarizedExperiment"
)
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) {
  stop("missing required R packages: ", paste(missing_packages, collapse = ", "), call. = FALSE)
}

normal_file <- function(path, label) {
  if (!file.exists(path) || dir.exists(path)) {
    stop(label, " does not exist or is not a file: ", path, call. = FALSE)
  }
  normalizePath(path, mustWork = TRUE)
}

args$spe <- normal_file(args$spe, "SPE RDS")
args$sce <- normal_file(args$sce, "SCE RDS")
args$sce_assays <- normal_file(args$sce_assays, "SCE HDF5 assay payload")
args$panel <- normal_file(args$panel, "gene panel")
args$protocol <- normal_file(args$protocol, "exploratory protocol")
args$output_root <- normalizePath(args$output_root, mustWork = FALSE)

sha256_file <- function(path) {
  executable <- Sys.which("sha256sum")
  if (!nzchar(executable)) {
    stop("sha256sum is required for provenance hashing", call. = FALSE)
  }
  output <- system2(executable, args = shQuote(path), stdout = TRUE, stderr = TRUE)
  status <- attr(output, "status")
  if ((!is.null(status) && status != 0L) || !length(output)) {
    stop("could not hash file: ", path, call. = FALSE)
  }
  value <- strsplit(output[[1]], "[[:space:]]+")[[1]][[1]]
  if (!grepl("^[0-9a-f]{64}$", value)) {
    stop("sha256sum returned an invalid digest for: ", path, call. = FALSE)
  }
  value
}

as_clean_character <- function(value, label) {
  value <- enc2utf8(as.character(value))
  if (anyNA(value) || any(!nzchar(trimws(value)))) {
    stop(label, " contains missing or empty values", call. = FALSE)
  }
  value
}

as_finite_number <- function(value, label, positive = FALSE, integral = FALSE) {
  if (is.factor(value)) value <- as.character(value)
  value <- suppressWarnings(as.numeric(value))
  if (anyNA(value) || any(!is.finite(value)) || any(value < 0)) {
    stop(label, " must contain finite non-negative values", call. = FALSE)
  }
  if (positive && any(value <= 0)) {
    stop(label, " must contain strictly positive values", call. = FALSE)
  }
  if (integral && any(abs(value - round(value)) > 1e-8)) {
    stop(label, " must contain integral values", call. = FALSE)
  }
  value
}

required_column <- function(frame, candidates, label, transform = identity) {
  available <- colnames(frame)
  selected <- candidates[candidates %in% available]
  if (!length(selected)) {
    stop(label, " is missing; expected one of: ", paste(candidates, collapse = ", "), call. = FALSE)
  }
  value <- if (is.matrix(frame)) frame[, selected[[1]]] else frame[[selected[[1]]]]
  list(value = transform(value), column = selected[[1]])
}

optional_column <- function(frame, candidates, transform = identity) {
  selected <- candidates[candidates %in% colnames(frame)]
  if (!length(selected)) return(NULL)
  value <- if (is.matrix(frame)) frame[, selected[[1]]] else frame[[selected[[1]]]]
  list(value = transform(value), column = selected[[1]])
}

panel_json <- jsonlite::fromJSON(args$panel, simplifyVector = FALSE)
panel_genes <- as_clean_character(unlist(panel_json$gene_ids, use.names = FALSE), "panel gene_ids")
if (anyDuplicated(panel_genes)) stop("frozen panel gene_ids must be unique", call. = FALSE)
if (!identical(panel_json$mode, "external_frozen")) {
  stop("panel mode must be external_frozen", call. = FALSE)
}
protocol_sha256 <- sha256_file(args$protocol)
if (!identical(protocol_sha256, frozen_protocol_sha256)) {
  stop(
    "protocol checksum differs from the frozen exploratory protocol: expected ",
    frozen_protocol_sha256, ", found ", protocol_sha256, call. = FALSE
  )
}
protocol_json <- jsonlite::fromJSON(args$protocol, simplifyVector = FALSE)
if (!identical(protocol_json$schema, "heir.spatialdlpfc_exploratory_protocol.v2") ||
    !identical(protocol_json$analysis_status, "exploratory_protocol_deviating_nonconfirmatory")) {
  stop("protocol schema or analysis_status is invalid", call. = FALSE)
}

resolve_protocol_path <- function(path) {
  if (grepl("^/", path)) normalizePath(path, mustWork = TRUE) else
    normalizePath(file.path(repo_root, path), mustWork = TRUE)
}

verify_frozen_input <- function(label, actual_path, specification, path_field, sha_field) {
  expected_path <- resolve_protocol_path(as.character(specification[[path_field]]))
  if (!identical(actual_path, expected_path)) {
    stop(label, " path is not the path frozen in the protocol", call. = FALSE)
  }
  message("Hashing frozen ", label, " before object access")
  actual_sha256 <- sha256_file(actual_path)
  expected_sha256 <- as.character(specification[[sha_field]])
  if (!identical(actual_sha256, expected_sha256)) {
    stop(
      label, " checksum differs from the frozen protocol: expected ", expected_sha256,
      ", found ", actual_sha256, call. = FALSE
    )
  }
  actual_sha256
}

immutable <- protocol_json$immutable_inputs
expected_panel_size <- as.integer(immutable$frozen_gene_panel$size)
if (length(panel_genes) != expected_panel_size ||
    as.integer(panel_json$gene_count) != expected_panel_size) {
  stop("frozen panel size differs from protocol immutable_inputs", call. = FALSE)
}
input_hashes <- list(
  spe = verify_frozen_input(
    "SPE", args$spe, immutable$spatial_experiment, "path", "sha256"
  ),
  sce = verify_frozen_input(
    "SCE metadata", args$sce, immutable$single_nucleus_experiment,
    "metadata_path", "metadata_sha256"
  ),
  sce_assays = verify_frozen_input(
    "SCE assay", args$sce_assays, immutable$single_nucleus_experiment,
    "assay_path", "assay_sha256"
  ),
  panel = verify_frozen_input(
    "gene panel", args$panel, immutable$frozen_gene_panel, "path", "sha256"
  ),
  protocol = protocol_sha256
)
if (!identical(
  as.character(panel_json$artifact_sha256),
  as.character(immutable$frozen_gene_panel$identity_sha256)
)) {
  stop("panel embedded identity differs from the protocol", call. = FALSE)
}

# The serialized HDF5Array seeds intentionally use the sibling basename
# "assays.h5".  Resolve that immutable relative seed from the extracted SCE
# directory; all other paths were normalized to absolute paths above.
setwd(dirname(args$sce))
message("Loading processed SpatialExperiment and HDF5-backed snRNA object")
spe <- readRDS(args$spe)
sce <- readRDS(args$sce)
if (!methods::is(spe, "SpatialExperiment")) {
  stop("--spe is not a SpatialExperiment", call. = FALSE)
}
if (!methods::is(sce, "SingleCellExperiment")) {
  stop("--sce is not a SingleCellExperiment", call. = FALSE)
}
if (!("counts" %in% SummarizedExperiment::assayNames(spe)) ||
    !("counts" %in% SummarizedExperiment::assayNames(sce))) {
  stop("both source objects must contain a raw counts assay", call. = FALSE)
}
if (nrow(spe) != as.integer(immutable$spatial_experiment$expected_genes) ||
    ncol(spe) != as.integer(immutable$spatial_experiment$expected_spots)) {
  stop("SPE dimensions differ from the frozen protocol", call. = FALSE)
}
if (nrow(sce) != as.integer(immutable$single_nucleus_experiment$expected_genes) ||
    ncol(sce) != as.integer(immutable$single_nucleus_experiment$expected_cells)) {
  stop("SCE dimensions differ from the frozen protocol", call. = FALSE)
}
sce_counts <- SummarizedExperiment::assay(sce, "counts")
if (!methods::is(sce_counts, "DelayedMatrix") ||
    !methods::is(DelayedArray::seed(sce_counts), "HDF5ArraySeed")) {
  stop("SCE counts assay must remain HDF5-backed", call. = FALSE)
}
sce_count_path <- normalizePath(HDF5Array::path(sce_counts), mustWork = TRUE)
if (!identical(sce_count_path, args$sce_assays)) {
  stop("SCE counts seed does not point to the declared --sce-assays payload", call. = FALSE)
}

map_panel <- function(object, object_label) {
  rows <- SummarizedExperiment::rowData(object)
  if (!("gene_name" %in% colnames(rows))) {
    stop(object_label, " rowData lacks required gene_name", call. = FALSE)
  }
  gene_names <- as_clean_character(rows[["gene_name"]], paste(object_label, "gene_name"))
  selected_counts <- table(factor(gene_names[gene_names %in% panel_genes], levels = panel_genes))
  if (any(selected_counts != 1L)) {
    offenders <- panel_genes[selected_counts != 1L]
    stop(
      object_label, " does not map the frozen panel one-to-one by gene_name: ",
      paste(head(offenders, 20L), collapse = ", "), call. = FALSE
    )
  }
  match(panel_genes, gene_names)
}

spe_panel_rows <- map_panel(spe, "SPE")
sce_panel_rows <- map_panel(sce, "SCE")

spot_meta <- SummarizedExperiment::colData(spe)
spot_sample <- required_column(
  spot_meta, c("sample_id"), "spot sample_id", function(x) as_clean_character(x, "spot sample_id")
)
spot_donor <- required_column(
  spot_meta, c("subject"), "spot donor", function(x) as_clean_character(x, "spot donor")
)
spot_position <- required_column(
  spot_meta, c("position"), "spot position", function(x) as_clean_character(x, "spot position")
)
normalize_position <- function(value, label) {
  normalized <- tolower(trimws(as.character(value)))
  normalized[normalized == "anterior"] <- "ant"
  normalized[normalized == "middle"] <- "mid"
  normalized[normalized == "posterior"] <- "post"
  if (any(!(normalized %in% c("ant", "mid", "post")))) {
    stop(label, " is outside the frozen ant/mid/post vocabulary", call. = FALSE)
  }
  normalized
}
spot_position$value <- normalize_position(spot_position$value, "spot position")
spot_library <- required_column(
  spot_meta, c("sum_umi"), "spot full-library UMI count",
  function(x) as_finite_number(x, "spot full-library UMI count", positive = TRUE, integral = TRUE)
)
spot_barcode_col <- optional_column(
  spot_meta, c("barcode", "array_barcode"),
  function(x) as_clean_character(x, "spot barcode")
)
spot_original_id <- as_clean_character(colnames(spe), "SPE column names")
spot_barcode <- if (is.null(spot_barcode_col)) spot_original_id else spot_barcode_col$value
spot_section <- spot_sample$value
spot_id <- spot_original_id
spot_id_rule <- "SPE_colname"
if (anyDuplicated(spot_id)) {
  spot_id <- paste(spot_section, spot_original_id, sep = ":")
  spot_id_rule <- "sample_id_colon_SPE_colname"
}
if (anyDuplicated(spot_id)) stop("spot IDs are not unique", call. = FALSE)

spatial <- SpatialExperiment::spatialCoords(spe)
spatial_or_coldata_column <- function(candidates, label, transform) {
  if (any(candidates %in% colnames(spatial))) {
    result <- required_column(spatial, candidates, label, transform)
    result$column <- paste0("spatialCoords.", result$column)
    return(result)
  }
  result <- required_column(spot_meta, candidates, label, transform)
  result$column <- paste0("colData.", result$column)
  result
}
x_coord <- spatial_or_coldata_column(
  c("pxl_col_in_fullres"), "full-resolution spot x",
  function(x) as_finite_number(x, "full-resolution spot x")
)
y_coord <- spatial_or_coldata_column(
  c("pxl_row_in_fullres"), "full-resolution spot y",
  function(x) as_finite_number(x, "full-resolution spot y")
)
array_row <- spatial_or_coldata_column(
  c("array_row"), "Visium array row",
  function(x) as_finite_number(x, "Visium array row", integral = TRUE)
)
array_col <- spatial_or_coldata_column(
  c("array_col"), "Visium array column",
  function(x) as_finite_number(x, "Visium array column", integral = TRUE)
)

reference_meta <- SummarizedExperiment::colData(sce)
reference_donor <- required_column(
  reference_meta, c("BrNum"), "reference donor",
  function(x) as_clean_character(x, "reference donor")
)
reference_sample <- required_column(
  reference_meta, c("Sample"), "reference sample_id",
  function(x) as_clean_character(x, "reference sample_id")
)
reference_position <- required_column(
  reference_meta, c("pos", "Position"), "reference position",
  function(x) as_clean_character(x, "reference position")
)
reference_position$value <- normalize_position(reference_position$value, "reference position")
reference_type <- required_column(
  reference_meta, c("cellType_broad_k"), "reference broad cell type",
  function(x) as_clean_character(x, "reference broad cell type")
)
reference_library <- required_column(
  reference_meta, c("sum"), "reference full-library UMI count",
  function(x) as_finite_number(x, "reference full-library UMI count", positive = TRUE, integral = TRUE)
)
reference_keep <- tolower(trimws(reference_type$value)) != "drop"
if (!any(reference_keep) || !any(!reference_keep)) {
  stop("reference cell_type filter must retain cells and exclude at least one 'drop' row", call. = FALSE)
}
expected_reference_sample <- paste(reference_donor$value, reference_position$value, sep = "_")
if (!identical(reference_sample$value, expected_reference_sample)) {
  stop("reference Sample is not the deterministic BrNum_position biological-bank ID", call. = FALSE)
}
protocol_reference_samples <- vapply(
  protocol_json$selected_queries, function(query) as.character(query$reference_sample), character(1)
)
normalized_protocol_samples <- sub("^snRNA_", "", protocol_reference_samples)
if (anyDuplicated(protocol_reference_samples) ||
    !all(normalized_protocol_samples %in% unique(reference_sample$value))) {
  stop("protocol reference_sample values do not map to biological SCE Sample banks", call. = FALSE)
}
reference_columns <- which(reference_keep)
reference_original_id <- as_clean_character(colnames(sce)[reference_keep], "retained SCE column names")
reference_id <- reference_original_id
reference_id_rule <- "SCE_colname"
if (anyDuplicated(reference_id)) {
  reference_id <- paste(reference_sample$value[reference_keep], reference_original_id, sep = ":")
  reference_id_rule <- "sample_id_colon_SCE_colname"
}
if (anyDuplicated(reference_id)) stop("reference IDs are not unique", call. = FALSE)

# The processed SPE uses BrNum_position sample IDs, whereas the frozen query
# protocol records the corresponding external Visium capture IDs.  Relabel only
# the ten prespecified query sections by donor+region; retain the actual sample_id
# separately and leave all other section IDs unchanged.
protocol_query_resolutions <- lapply(protocol_json$selected_queries, function(query) {
  donor <- as.character(query$donor)
  position <- normalize_position(as.character(query$position), "protocol query position")
  processed <- unique(spot_sample$value[
    spot_donor$value == donor & spot_position$value == position
  ])
  if (length(processed) != 1L) {
    stop("protocol query does not resolve to exactly one processed SPE sample", call. = FALSE)
  }
  list(
    donor = donor, position = position, processed_sample_id = processed[[1]],
    protocol_section = as.character(query$section)
  )
})
for (resolution in protocol_query_resolutions) {
  rows <- spot_sample$value == resolution$processed_sample_id
  spot_section[rows] <- resolution$protocol_section
}
if (anyDuplicated(unique(spot_section)) || length(unique(spot_section)) != 30L) {
  stop("protocol section relabeling did not preserve 30 unique SPE sections", call. = FALSE)
}

image_data <- SpatialExperiment::imgData(spe)
for (column in c("sample_id", "image_id", "data", "scaleFactor")) {
  if (!(column %in% colnames(image_data))) stop("imgData lacks required column: ", column, call. = FALSE)
}
image_ids <- as_clean_character(image_data[["image_id"]], "imgData image_id")
hires_rows <- which(tolower(image_ids) == "hires")
if (length(hires_rows) != 30L) {
  stop("expected exactly 30 embedded hires images, found ", length(hires_rows), call. = FALSE)
}
image_sample_ids <- as_clean_character(
  image_data[["sample_id"]][hires_rows], "hires image sample_id"
)
processed_sample_ids <- unique(spot_sample$value)
section_by_sample <- setNames(vapply(processed_sample_ids, function(sample_id) {
  values <- unique(spot_section[spot_sample$value == sample_id])
  if (length(values) != 1L) stop("processed sample maps to multiple section IDs", call. = FALSE)
  values[[1]]
}, character(1)), processed_sample_ids)
image_sections <- unname(section_by_sample[image_sample_ids])
if (anyNA(image_sections) || anyDuplicated(image_sections)) {
  stop("hires images are not one-to-one with exported sections", call. = FALSE)
}
if (!setequal(unique(spot_section), image_sections) || length(unique(spot_section)) != 30L) {
  stop("spot sections and embedded hires image sections are not the same 30 IDs", call. = FALSE)
}
if (length(unique(spot_donor$value)) != as.integer(immutable$spatial_experiment$expected_donors) ||
    length(unique(spot_section)) != as.integer(immutable$spatial_experiment$expected_sections)) {
  stop("SPE donor/section counts differ from the frozen protocol", call. = FALSE)
}
protocol_sections <- vapply(
  protocol_json$selected_queries, function(query) as.character(query$section), character(1)
)
if (anyDuplicated(protocol_sections) || !all(protocol_sections %in% unique(spot_section))) {
  stop("protocol selected query sections are not present in the SPE", call. = FALSE)
}
image_scales <- as_finite_number(
  image_data[["scaleFactor"]][hires_rows], "hires image scaleFactor", positive = TRUE
)

if (args$preflight) {
  cat(jsonlite::toJSON(list(
    status = "preflight_passed_no_outputs_written",
    protocol_sha256 = protocol_sha256,
    input_sha256 = input_hashes,
    panel_genes = length(panel_genes),
    spots = ncol(spe),
    spot_donors = length(unique(spot_donor$value)),
    spot_sections = length(unique(spot_section)),
    reference_cells = ncol(sce),
    reference_biological_banks = length(unique(reference_sample$value)),
    reference_drop_rows = sum(!reference_keep),
    embedded_hires_images = length(hires_rows),
    protocol_query_resolutions = protocol_query_resolutions
  ), auto_unbox = TRUE, pretty = TRUE), "\n")
  quit(status = 0L)
}

source_path <- file.path(args$output_root, "source.h5")
receipt_path <- file.path(args$output_root, "receipt.json")
image_root <- file.path(args$output_root, "images")
if (!args$overwrite && (file.exists(source_path) || file.exists(receipt_path))) {
  stop("completed output already exists; pass --overwrite to replace it", call. = FALSE)
}
dir.create(image_root, recursive = TRUE, showWarnings = FALSE)
if (!dir.exists(image_root)) stop("could not create output directory: ", image_root, call. = FALSE)

safe_name <- function(value) {
  result <- gsub("[^A-Za-z0-9._-]+", "_", value)
  if (!nzchar(result)) stop("section cannot be converted to a safe image filename", call. = FALSE)
  result
}

expected_image_paths <- file.path(
  image_root, paste0(vapply(image_sections, safe_name, character(1)), "_hires.png")
)
if (!args$overwrite && any(file.exists(expected_image_paths))) {
  stop("one or more image outputs already exist; pass --overwrite to replace them", call. = FALSE)
}

message("Exporting 30 embedded hires images")
image_records <- vector("list", length(hires_rows))
for (j in seq_along(hires_rows)) {
  row <- hires_rows[[j]]
  section <- image_sections[[j]]
  output_path <- normalizePath(
    file.path(image_root, paste0(safe_name(section), "_hires.png")), mustWork = FALSE
  )
  temporary_path <- paste0(output_path, ".partial-", Sys.getpid())
  raster <- SpatialExperiment::imgRaster(image_data[["data"]][[row]])
  image <- magick::image_read(raster)
  info <- magick::image_info(image)
  if (nrow(info) != 1L || info$width <= 0L || info$height <= 0L) {
    stop("invalid hires image for section: ", section, call. = FALSE)
  }
  magick::image_write(image, path = temporary_path, format = "png")
  if (file.exists(output_path) && args$overwrite) unlink(output_path)
  if (!file.rename(temporary_path, output_path)) {
    stop("could not atomically install image: ", output_path, call. = FALSE)
  }
  image_records[[j]] <- list(
    section = section, path = output_path, width = as.integer(info$width),
    height = as.integer(info$height), scale_factor = image_scales[[j]]
  )
  rm(raster, image)
}

write_character <- function(file, path, value) {
  value <- as_clean_character(value, path)
  size <- max(1L, max(nchar(value, type = "bytes")))
  rhdf5::h5createDataset(
    file, path, dims = length(value), storage.mode = "character", size = size,
    encoding = "UTF-8", chunk = min(length(value), 4096L), level = 4L
  )
  rhdf5::h5write(value, file, path)
}

write_vector <- function(file, path, value, mode) {
  value <- switch(
    mode,
    integer = as.integer(value),
    double = as.numeric(value),
    stop("unsupported HDF5 vector mode", call. = FALSE)
  )
  rhdf5::h5createDataset(
    file, path, dims = length(value), storage.mode = mode,
    chunk = min(length(value), 16384L), level = 4L
  )
  rhdf5::h5write(value, file, path)
}

create_count_dataset <- function(file, path, observations, genes, block_size) {
  # rhdf5 reverses R dimensions in the physical HDF5 layout.  The R view is
  # genes x observations so h5py and other row-major readers see observations x genes.
  rhdf5::h5createDataset(
    file, path, dims = c(genes, observations), storage.mode = "integer",
    chunk = c(genes, min(block_size, observations)), level = 4L, shuffle = TRUE
  )
}

write_count_blocks <- function(
  matrix, panel_rows, observation_columns, libraries, file, path, block_size, label
) {
  n_observations <- length(observation_columns)
  n_genes <- length(panel_rows)
  create_count_dataset(file, path, n_observations, n_genes, block_size)
  starts <- seq.int(1L, n_observations, by = block_size)
  for (start in starts) {
    stop_at <- min(n_observations, start + block_size - 1L)
    local <- start:stop_at
    source_columns <- observation_columns[local]
    block <- as.matrix(matrix[panel_rows, source_columns, drop = FALSE])
    if (!is.numeric(block) || anyNA(block) || any(!is.finite(block)) || any(block < 0) ||
        any(abs(block - round(block)) > 1e-8) || any(block > .Machine$integer.max)) {
      stop(label, " panel counts are not finite non-negative int32 UMIs", call. = FALSE)
    }
    panel_library <- colSums(block)
    if (any(panel_library > libraries[local] + 1e-8)) {
      stop(label, " panel count sum exceeds the full-library UMI count", call. = FALSE)
    }
    storage.mode(block) <- "integer"
    rhdf5::h5write(block, file, path, index = list(seq_len(n_genes), local))
    rm(block, panel_library)
    if ((start - 1L) %% (block_size * 8L) == 0L) gc(verbose = FALSE)
  }
}

temporary_source <- paste0(source_path, ".partial-", Sys.getpid())
if (file.exists(temporary_source)) unlink(temporary_source)
if (!rhdf5::h5createFile(temporary_source)) stop("could not create temporary HDF5 source", call. = FALSE)
for (group in c("panel", "spots", "reference", "images", "provenance")) {
  rhdf5::h5createGroup(temporary_source, group)
}

write_character(temporary_source, "/panel/genes", panel_genes)
write_character(temporary_source, "/spots/id", spot_id)
write_character(temporary_source, "/spots/barcode", spot_barcode)
write_character(temporary_source, "/spots/donor", spot_donor$value)
write_character(temporary_source, "/spots/section", spot_section)
write_character(temporary_source, "/spots/sample_id", spot_sample$value)
write_character(temporary_source, "/spots/position", spot_position$value)
write_vector(temporary_source, "/spots/x_fullres", x_coord$value, "double")
write_vector(temporary_source, "/spots/y_fullres", y_coord$value, "double")
write_vector(temporary_source, "/spots/array_row", array_row$value, "integer")
write_vector(temporary_source, "/spots/array_col", array_col$value, "integer")
write_vector(temporary_source, "/spots/library", spot_library$value, "double")

write_character(temporary_source, "/reference/id", reference_id)
write_character(temporary_source, "/reference/donor", reference_donor$value[reference_keep])
write_character(temporary_source, "/reference/sample_id", reference_sample$value[reference_keep])
write_character(temporary_source, "/reference/position", reference_position$value[reference_keep])
write_character(temporary_source, "/reference/cell_type", reference_type$value[reference_keep])
write_vector(
  temporary_source, "/reference/library", reference_library$value[reference_keep], "double"
)

write_character(
  temporary_source, "/images/section", vapply(image_records, `[[`, character(1), "section")
)
write_character(
  temporary_source, "/images/path", vapply(image_records, `[[`, character(1), "path")
)
write_vector(
  temporary_source, "/images/width", vapply(image_records, `[[`, integer(1), "width"), "integer"
)
write_vector(
  temporary_source, "/images/height", vapply(image_records, `[[`, integer(1), "height"), "integer"
)
write_vector(
  temporary_source, "/images/scale_factor",
  vapply(image_records, `[[`, numeric(1), "scale_factor"), "double"
)

metadata_columns <- list(
  spots = list(
    donor = spot_donor$column, section = spot_sample$column,
    sample_id = spot_sample$column, position = spot_position$column,
    barcode = if (is.null(spot_barcode_col)) "SPE_colname" else spot_barcode_col$column,
    library = spot_library$column, x_fullres = x_coord$column, y_fullres = y_coord$column,
    array_row = array_row$column, array_col = array_col$column, id_rule = spot_id_rule,
    position_normalization = "anterior_middle_posterior_to_ant_mid_post",
    section_rule = "sample_id_except_prespecified_queries_relabelled_by_protocol_donor_position"
  ),
  reference = list(
    donor = reference_donor$column, sample_id = reference_sample$column,
    position = reference_position$column, cell_type = reference_type$column,
    library = reference_library$column, id_rule = reference_id_rule
  )
)
input_identity <- list(
  spe = list(
    path = args$spe, bytes = unname(file.info(args$spe)$size), sha256 = input_hashes$spe
  ),
  sce = list(
    path = args$sce, bytes = unname(file.info(args$sce)$size), sha256 = input_hashes$sce
  ),
  sce_assays = list(
    path = args$sce_assays, bytes = unname(file.info(args$sce_assays)$size),
    sha256 = input_hashes$sce_assays
  ),
  panel = list(
    path = args$panel, bytes = unname(file.info(args$panel)$size), sha256 = input_hashes$panel
  ),
  protocol = list(
    path = args$protocol, bytes = unname(file.info(args$protocol)$size),
    sha256 = input_hashes$protocol
  )
)
provenance <- c(
  schema = "heir.spatialdlpfc_exploratory_source.v2",
  analysis_status = "exploratory_protocol_deviating_nonconfirmatory",
  scientific_scope = "regional_mechanism_exploration_only_not_registered_cell_level_validation",
  count_assay = "raw_counts_only",
  panel_mapping = "exact_one_to_one_gene_name_in_both_objects",
  image_source = "SpatialExperiment_embedded_hires",
  image_qualification = "downsampled_not_original_full_resolution_not_0.5_um_per_pixel",
  hoptimus_use_boundary = "mesoscopic_regional_proxy_only",
  reference_sample_identity = "Sample_equals_BrNum_underscore_pos",
  protocol_reference_normalization = "strip_leading_snRNA_prefix_before_matching_to_Sample",
  protocol_query_resolutions_json = jsonlite::toJSON(
    protocol_query_resolutions, auto_unbox = TRUE
  ),
  panel_embedded_artifact_sha256 = as.character(panel_json$artifact_sha256),
  protocol_sha256 = protocol_sha256,
  predecessor_protocol_sha256 = as.character(protocol_json$revision$predecessor_sha256),
  protocol_revision_json = jsonlite::toJSON(protocol_json$revision, auto_unbox = TRUE),
  panel_compatibility_json = jsonlite::toJSON(list(
    source_panel_sha256 = panel_json$source_panel_sha256,
    source_panel_identity_sha256 = panel_json$source_panel_identity_sha256,
    excluded_unmappable_gene_ids = panel_json$excluded_unmappable_gene_ids,
    compatibility_rule = panel_json$compatibility_rule
  ), auto_unbox = TRUE),
  metadata_columns_json = jsonlite::toJSON(metadata_columns, auto_unbox = TRUE),
  input_identity_json = jsonlite::toJSON(input_identity, auto_unbox = TRUE),
  created_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)
)
for (name in names(provenance)) {
  write_character(temporary_source, paste0("/provenance/", name), provenance[[name]])
}

message("Writing raw panel counts in bounded observation blocks")
write_count_blocks(
  SummarizedExperiment::assay(spe, "counts"), spe_panel_rows, seq_len(ncol(spe)),
  spot_library$value, temporary_source, "/spots/counts", args$block_size, "spot"
)
write_count_blocks(
  SummarizedExperiment::assay(sce, "counts"), sce_panel_rows, reference_columns,
  reference_library$value[reference_keep], temporary_source, "/reference/counts",
  args$block_size, "reference"
)
rhdf5::h5closeAll()
if (file.exists(source_path) && args$overwrite) unlink(source_path)
if (!file.rename(temporary_source, source_path)) {
  stop("could not atomically install HDF5 source: ", source_path, call. = FALSE)
}

message("Hashing completed outputs for the receipt")
for (j in seq_along(image_records)) {
  image_records[[j]]$bytes <- unname(file.info(image_records[[j]]$path)$size)
  image_records[[j]]$sha256 <- sha256_file(image_records[[j]]$path)
}

receipt <- list(
  schema = "heir.spatialdlpfc_exploratory_source_receipt.v2",
  analysis_status = "exploratory_protocol_deviating_nonconfirmatory",
  created_utc = provenance[["created_utc"]],
  source = list(
    path = source_path, bytes = unname(file.info(source_path)$size),
    sha256 = sha256_file(source_path), physical_count_axis_order = c("observations", "genes")
  ),
  inputs = input_identity,
  panel = list(
    genes = panel_genes, gene_count = length(panel_genes),
    mapping = "exact_one_to_one_gene_name_in_both_objects",
    embedded_artifact_sha256 = panel_json$artifact_sha256,
    source_panel_sha256 = panel_json$source_panel_sha256,
    excluded_unmappable_gene_ids = panel_json$excluded_unmappable_gene_ids,
    compatibility_rule = panel_json$compatibility_rule
  ),
  protocol_revision = protocol_json$revision,
  protocol_query_resolutions = protocol_query_resolutions,
  observations = list(
    spots = ncol(spe), reference_retained = sum(reference_keep),
    reference_drop_excluded = sum(!reference_keep), sections = length(image_records)
  ),
  metadata_columns = metadata_columns,
  images = image_records,
  validations = list(
    raw_counts_assay_only = TRUE, count_dtype = "int32", counts_nonnegative_integral = TRUE,
    panel_sum_not_greater_than_full_library = TRUE,
    spot_count_shape = c(ncol(spe), length(panel_genes)),
    reference_count_shape = c(sum(reference_keep), length(panel_genes)),
    embedded_hires_images = 30L
  ),
  limitations = c(
    "processed_target_was_opened_before_this_exploratory_run",
    "embedded_hires_images_are_downsampled_and_not_original_full_resolution",
    "images_are_not_qualified_at_H_optimus_1_native_0.5_um_per_pixel",
    "H_optimus_1_features_from_these_images_are_mesoscopic_regional_proxies_only",
    "no_registered_cell_level_claim_or_ST_floor_is_supported"
  )
)
temporary_receipt <- paste0(receipt_path, ".partial-", Sys.getpid())
jsonlite::write_json(receipt, temporary_receipt, auto_unbox = TRUE, pretty = TRUE, na = "null")
if (file.exists(receipt_path) && args$overwrite) unlink(receipt_path)
if (!file.rename(temporary_receipt, receipt_path)) {
  stop("could not atomically install receipt: ", receipt_path, call. = FALSE)
}

message("Wrote exploratory spatialDLPFC source: ", source_path)
message("Wrote provenance receipt: ", receipt_path)
