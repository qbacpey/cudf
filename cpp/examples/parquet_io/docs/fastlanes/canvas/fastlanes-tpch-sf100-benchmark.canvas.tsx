import {
  BarChart,
  Callout,
  Code,
  CollapsibleSection,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
  useState,
} from "cursor/canvas";

type Codec = "NONE" | "SNAPPY" | "ZSTD";
type Enc = "PLAIN" | "DICT" | "DELTA" | "BSS" | "FastLanes";
type ColumnInfo = { name: string; type: string; rows: number; encoder: "NATIVE64" | "RAW32" };
type AggRow = { label: string; gb: number; bits: number; write: number; read: number; fastlanesColumns?: number };
type PlanTotal = { plan: string; gb: number; rewriteS: number; readS: number };
type TableRow = { table: string; codec: "SNAPPY" | "ZSTD"; plans: { mb: number; rewriteS: number; readS: number }[] };
type AblationRow = { column: string; encoding: string; rowsPerPage: number; bits: number; write: number; read: number };

const CODECS: Codec[] = ["NONE", "SNAPPY", "ZSTD"];
const STANDARD: Enc[] = ["PLAIN", "DICT", "DELTA", "BSS"];

const COLUMNS: ColumnInfo[] = [
  { name: "lineitem.l_orderkey", type: "INT64", rows: 600037902, encoder: "NATIVE64" },
  { name: "lineitem.l_partkey", type: "INT64", rows: 600037902, encoder: "NATIVE64" },
  { name: "lineitem.l_suppkey", type: "INT64", rows: 600037902, encoder: "NATIVE64" },
  { name: "lineitem.l_linenumber", type: "INT64", rows: 600037902, encoder: "NATIVE64" },
  { name: "lineitem.l_shipdate", type: "date", rows: 600037902, encoder: "RAW32" },
  { name: "lineitem.l_commitdate", type: "date", rows: 600037902, encoder: "RAW32" },
  { name: "lineitem.l_receiptdate", type: "date", rows: 600037902, encoder: "RAW32" },
  { name: "orders.o_orderkey", type: "INT64", rows: 150000000, encoder: "NATIVE64" },
  { name: "orders.o_custkey", type: "INT64", rows: 150000000, encoder: "NATIVE64" },
  { name: "orders.o_orderdate", type: "date", rows: 150000000, encoder: "RAW32" },
  { name: "orders.o_shippriority", type: "INT32", rows: 150000000, encoder: "RAW32" },
  { name: "partsupp.ps_partkey", type: "INT64", rows: 80000000, encoder: "NATIVE64" },
  { name: "partsupp.ps_suppkey", type: "INT64", rows: 80000000, encoder: "NATIVE64" },
  { name: "partsupp.ps_availqty", type: "INT64", rows: 80000000, encoder: "NATIVE64" },
  { name: "part.p_partkey", type: "INT64", rows: 20000000, encoder: "NATIVE64" },
  { name: "part.p_size", type: "INT32", rows: 20000000, encoder: "RAW32" },
  { name: "customer.c_custkey", type: "INT64", rows: 15000000, encoder: "NATIVE64" },
  { name: "customer.c_nationkey", type: "INT32", rows: 15000000, encoder: "RAW32" },
  { name: "supplier.s_suppkey", type: "INT64", rows: 1000000, encoder: "NATIVE64" },
  { name: "supplier.s_nationkey", type: "INT32", rows: 1000000, encoder: "RAW32" },
];

const BITS: Record<Codec, Record<Enc, (number | null)[]>> = {
  NONE: {
    PLAIN: [64.020, 64.020, 64.020, 64.020, 32.020, 32.020, 32.020, 64.020, 64.020, 32.020, 32.020, 64.020, 64.020, 64.020, 64.020, 32.019, 64.020, 32.020, 64.021, 32.021],
    DICT: [23.930, null, null, 3.041, 12.695, 12.680, 12.699, null, null, 12.665, 0.021, 22.022, null, 19.249, null, 6.050, null, 5.044, null, 5.046],
    DELTA: [5.053, 26.045, 21.461, 3.335, 12.468, 12.466, 12.468, 5.337, 25.524, 12.843, 0.334, 1.335, 20.461, 15.089, 0.335, 7.333, 0.335, 6.334, 0.336, 6.335],
    BSS: [64.020, 64.020, 64.020, 64.020, 32.020, 32.020, 32.020, 64.020, 64.020, 32.020, 32.020, 64.020, 64.020, 64.020, 64.020, 32.019, 64.020, 32.020, 64.021, 32.021],
    FastLanes: [15.070, 25.070, 20.070, 3.069, 12.069, 12.070, 12.070, 17.070, 24.070, 12.069, 1.069, 13.070, 20.070, 14.070, 15.070, 6.069, 15.070, 5.070, 15.078, 5.073],
  },
  SNAPPY: {
    PLAIN: [12.482, 43.683, 39.520, 7.203, 25.149, 24.839, 25.174, 31.910, 42.858, 26.273, 1.521, 14.040, 32.060, 30.076, 32.032, 15.732, 32.032, 15.407, 32.031, 15.408],
    DICT: [16.006, null, null, 2.208, 12.695, 12.680, 12.699, null, null, 12.665, 0.021, 14.077, null, 16.760, null, 6.050, null, 5.044, null, 5.046],
    DELTA: [2.664, 26.045, 21.461, 2.462, 12.468, 12.466, 12.468, 0.326, 25.524, 12.843, 0.037, 0.098, 1.064, 15.089, 0.040, 7.293, 0.041, 6.281, 0.041, 6.283],
    BSS: [7.504, 28.189, 25.836, 5.410, 13.201, 12.873, 13.234, 3.300, 26.084, 15.981, 1.521, 9.323, 5.091, 18.389, 5.375, 9.205, 5.375, 9.167, 5.376, 9.168],
    FastLanes: [9.157, 25.070, 20.070, 3.052, 12.069, 12.070, 12.070, 17.070, 24.070, 12.069, 0.074, 4.907, 20.070, 14.070, 15.070, 6.069, 15.069, 5.067, 15.073, 5.068],
  },
  ZSTD: {
    PLAIN: [2.661, 28.075, 23.922, 3.629, 15.767, 15.542, 15.786, 7.251, 25.076, 16.215, 0.031, 2.084, 8.188, 15.900, 8.341, 8.334, 8.341, 7.546, 8.346, 7.549],
    DICT: [11.344, null, null, 1.481, 12.306, 12.280, 12.313, null, null, 12.261, 0.021, 8.097, null, 15.349, null, 6.003, null, 4.998, null, 5.000],
    DELTA: [1.852, 25.974, 21.289, 1.634, 12.046, 12.030, 12.047, 0.036, 25.347, 12.781, 0.028, 0.033, 0.039, 15.000, 0.031, 7.107, 0.031, 5.956, 0.032, 5.958],
    BSS: [2.489, 25.455, 21.980, 1.631, 11.228, 10.976, 11.254, 0.249, 24.039, 12.738, 0.031, 0.467, 0.524, 14.724, 0.208, 5.775, 0.208, 4.807, 0.209, 4.808],
    FastLanes: [5.549, 25.033, 20.033, 2.796, 11.923, 11.911, 11.925, 16.256, 24.033, 11.910, 0.029, 3.112, 18.345, 14.002, 14.736, 5.982, 14.736, 4.975, 14.737, 4.977],
  },
};

type Speed = { write: number; read: number };
const SPEED: Record<Codec, Record<Enc, Speed[]>> = {
  NONE: {
    PLAIN: [{ write: 4.61, read: 10.54 }, { write: 4.73, read: 10.82 }, { write: 4.79, read: 10.89 }, { write: 4.69, read: 10.61 }, { write: 4.29, read: 9.77 }, { write: 4.44, read: 10.07 }, { write: 4.28, read: 9.68 }, { write: 4.91, read: 11.37 }, { write: 4.91, read: 11.36 }, { write: 4.48, read: 10.69 }, { write: 4.46, read: 10.40 }, { write: 4.90, read: 10.98 }, { write: 4.88, read: 11.08 }, { write: 5.10, read: 11.35 }, { write: 4.93, read: 11.45 }, { write: 4.54, read: 10.87 }, { write: 4.83, read: 11.01 }, { write: 4.36, read: 10.65 }, { write: 4.19, read: 5.71 }, { write: 3.18, read: 4.12 }],
    DICT: [{ write: 6.32, read: 19.03 }, { write: 0, read: 0 }, { write: 0, read: 0 }, { write: 19.84, read: 59.54 }, { write: 5.99, read: 16.30 }, { write: 6.00, read: 16.40 }, { write: 6.04, read: 16.22 }, { write: 0, read: 0 }, { write: 0, read: 0 }, { write: 6.21, read: 18.60 }, { write: 13.44, read: 70.26 }, { write: 6.17, read: 21.69 }, { write: 0, read: 0 }, { write: 9.02, read: 27.92 }, { write: 0, read: 0 }, { write: 8.92, read: 26.92 }, { write: 0, read: 0 }, { write: 8.83, read: 29.03 }, { write: 0, read: 0 }, { write: 3.41, read: 4.60 }],
    DELTA: [{ write: 22.66, read: 47.90 }, { write: 9.73, read: 20.52 }, { write: 10.88, read: 23.10 }, { write: 25.79, read: 51.88 }, { write: 8.19, read: 17.34 }, { write: 7.99, read: 15.94 }, { write: 7.96, read: 16.03 }, { write: 23.77, read: 56.66 }, { write: 10.00, read: 22.74 }, { write: 8.20, read: 17.68 }, { write: 19.50, read: 45.15 }, { write: 33.63, read: 80.79 }, { write: 11.78, read: 26.66 }, { write: 14.93, read: 32.47 }, { write: 37.24, read: 98.43 }, { write: 10.96, read: 24.90 }, { write: 33.58, read: 83.07 }, { write: 11.32, read: 24.46 }, { write: 11.06, read: 8.44 }, { write: 4.48, read: 3.86 }],
    BSS: [{ write: 4.81, read: 10.32 }, { write: 4.86, read: 10.37 }, { write: 4.97, read: 10.68 }, { write: 4.77, read: 10.21 }, { write: 4.34, read: 9.54 }, { write: 4.26, read: 9.25 }, { write: 4.28, read: 9.10 }, { write: 4.92, read: 10.92 }, { write: 4.90, read: 10.48 }, { write: 4.49, read: 10.19 }, { write: 4.45, read: 10.24 }, { write: 5.01, read: 10.65 }, { write: 4.93, read: 10.84 }, { write: 5.01, read: 11.03 }, { write: 4.98, read: 11.05 }, { write: 4.52, read: 10.43 }, { write: 4.95, read: 10.79 }, { write: 4.42, read: 10.69 }, { write: 4.03, read: 5.18 }, { write: 3.24, read: 4.38 }],
    FastLanes: [{ write: 1.14, read: 30.62 }, { write: 1.13, read: 21.59 }, { write: 1.15, read: 26.08 }, { write: 1.23, read: 64.47 }, { write: 0.71, read: 19.53 }, { write: 0.71, read: 18.45 }, { write: 0.71, read: 19.07 }, { write: 1.19, read: 30.99 }, { write: 1.15, read: 23.99 }, { write: 0.72, read: 22.65 }, { write: 0.76, read: 54.04 }, { write: 1.20, read: 36.75 }, { write: 1.17, read: 27.05 }, { write: 1.20, read: 35.45 }, { write: 1.21, read: 32.96 }, { write: 0.80, read: 35.95 }, { write: 1.24, read: 33.56 }, { write: 0.77, read: 33.09 }, { write: 1.50, read: 8.79 }, { write: 0.80, read: 7.22 }],
  },
  SNAPPY: {
    PLAIN: [{ write: 7.42, read: 25.60 }, { write: 2.91, read: 12.37 }, { write: 3.05, read: 13.30 }, { write: 9.59, read: 36.53 }, { write: 2.23, read: 9.89 }, { write: 2.24, read: 10.19 }, { write: 2.24, read: 10.13 }, { write: 3.45, read: 16.35 }, { write: 2.86, read: 12.92 }, { write: 2.17, read: 10.52 }, { write: 11.65, read: 39.29 }, { write: 6.83, read: 25.65 }, { write: 3.37, read: 16.03 }, { write: 3.36, read: 16.94 }, { write: 3.22, read: 17.24 }, { write: 2.31, read: 15.56 }, { write: 2.79, read: 15.13 }, { write: 1.97, read: 14.42 }, { write: 0.59, read: 3.95 }, { write: 0.36, read: 2.86 }],
    DICT: [{ write: 2.72, read: 20.04 }, { write: 0, read: 0 }, { write: 0, read: 0 }, { write: 16.85, read: 58.23 }, { write: 5.44, read: 15.91 }, { write: 5.41, read: 16.48 }, { write: 5.44, read: 16.26 }, { write: 0, read: 0 }, { write: 0, read: 0 }, { write: 5.57, read: 17.66 }, { write: 12.82, read: 67.85 }, { write: 2.63, read: 22.15 }, { write: 0, read: 0 }, { write: 5.49, read: 28.51 }, { write: 0, read: 0 }, { write: 8.30, read: 25.76 }, { write: 0, read: 0 }, { write: 8.28, read: 29.05 }, { write: 0, read: 0 }, { write: 2.84, read: 5.41 }],
    DELTA: [{ write: 17.26, read: 49.37 }, { write: 8.30, read: 20.65 }, { write: 9.29, read: 22.90 }, { write: 21.00, read: 49.03 }, { write: 7.11, read: 16.81 }, { write: 7.01, read: 16.14 }, { write: 7.01, read: 15.85 }, { write: 31.98, read: 70.87 }, { write: 8.58, read: 22.77 }, { write: 7.17, read: 17.91 }, { write: 19.46, read: 41.89 }, { write: 35.66, read: 75.55 }, { write: 24.10, read: 64.92 }, { write: 12.70, read: 32.67 }, { write: 36.64, read: 96.80 }, { write: 9.89, read: 25.24 }, { write: 33.14, read: 81.87 }, { write: 9.68, read: 22.20 }, { write: 9.98, read: 6.71 }, { write: 3.05, read: 3.26 }],
    BSS: [{ write: 10.20, read: 29.11 }, { write: 6.39, read: 16.98 }, { write: 7.13, read: 17.94 }, { write: 10.97, read: 34.46 }, { write: 4.65, read: 15.08 }, { write: 4.73, read: 15.08 }, { write: 4.65, read: 14.77 }, { write: 15.02, read: 35.28 }, { write: 7.52, read: 19.85 }, { write: 4.71, read: 14.42 }, { write: 11.66, read: 36.88 }, { write: 12.10, read: 32.47 }, { write: 13.95, read: 39.22 }, { write: 9.11, read: 24.16 }, { write: 13.47, read: 39.16 }, { write: 7.74, read: 21.96 }, { write: 12.72, read: 39.68 }, { write: 6.92, read: 21.60 }, { write: 3.12, read: 6.13 }, { write: 2.23, read: 4.12 }],
    FastLanes: [{ write: 1.13, read: 35.00 }, { write: 1.11, read: 21.98 }, { write: 1.13, read: 24.90 }, { write: 1.23, read: 63.80 }, { write: 0.70, read: 20.07 }, { write: 0.70, read: 20.06 }, { write: 0.70, read: 19.52 }, { write: 1.17, read: 30.70 }, { write: 1.12, read: 24.02 }, { write: 0.71, read: 22.27 }, { write: 0.77, read: 61.48 }, { write: 1.22, read: 51.88 }, { write: 1.15, read: 27.97 }, { write: 1.19, read: 34.73 }, { write: 1.19, read: 31.83 }, { write: 0.80, read: 32.13 }, { write: 1.23, read: 32.14 }, { write: 0.76, read: 31.31 }, { write: 1.34, read: 8.68 }, { write: 0.83, read: 5.69 }],
  },
  ZSTD: {
    PLAIN: [{ write: 6.36, read: 27.47 }, { write: 2.13, read: 11.36 }, { write: 2.15, read: 12.80 }, { write: 5.93, read: 26.15 }, { write: 1.97, read: 9.49 }, { write: 1.99, read: 9.71 }, { write: 1.97, read: 9.46 }, { write: 2.96, read: 15.13 }, { write: 2.14, read: 14.03 }, { write: 1.93, read: 10.69 }, { write: 11.91, read: 48.34 }, { write: 5.21, read: 29.79 }, { write: 2.71, read: 15.67 }, { write: 2.36, read: 15.99 }, { write: 2.87, read: 16.43 }, { write: 2.17, read: 11.59 }, { write: 2.40, read: 13.29 }, { write: 1.77, read: 9.12 }, { write: 0.45, read: 1.52 }, { write: 0.30, read: 0.99 }],
    DICT: [{ write: 2.21, read: 13.15 }, { write: 0, read: 0 }, { write: 0, read: 0 }, { write: 14.83, read: 53.94 }, { write: 3.34, read: 11.87 }, { write: 3.34, read: 11.73 }, { write: 3.33, read: 12.28 }, { write: 0, read: 0 }, { write: 0, read: 0 }, { write: 3.41, read: 13.27 }, { write: 12.37, read: 60.40 }, { write: 2.20, read: 14.38 }, { write: 0, read: 0 }, { write: 3.54, read: 19.50 }, { write: 0, read: 0 }, { write: 6.02, read: 19.67 }, { write: 0, read: 0 }, { write: 5.92, read: 20.87 }, { write: 0, read: 0 }, { write: 2.01, read: 2.54 }],
    DELTA: [{ write: 15.23, read: 42.90 }, { write: 3.90, read: 14.07 }, { write: 4.69, read: 15.63 }, { write: 18.54, read: 48.68 }, { write: 4.06, read: 11.63 }, { write: 4.04, read: 11.08 }, { write: 4.02, read: 11.16 }, { write: 32.10, read: 81.71 }, { write: 4.06, read: 14.80 }, { write: 3.91, read: 12.09 }, { write: 18.05, read: 40.88 }, { write: 34.12, read: 81.79 }, { write: 26.73, read: 69.54 }, { write: 6.90, read: 21.95 }, { write: 34.70, read: 83.59 }, { write: 6.39, read: 16.71 }, { write: 31.59, read: 73.59 }, { write: 6.33, read: 16.27 }, { write: 9.61, read: 7.54 }, { write: 2.02, read: 2.09 }],
    BSS: [{ write: 9.20, read: 34.87 }, { write: 3.45, read: 15.00 }, { write: 3.92, read: 15.48 }, { write: 9.96, read: 36.82 }, { write: 3.27, read: 12.16 }, { write: 3.32, read: 12.20 }, { write: 3.27, read: 12.06 }, { write: 15.11, read: 53.25 }, { write: 3.93, read: 18.77 }, { write: 3.19, read: 13.32 }, { write: 11.95, read: 43.11 }, { write: 16.06, read: 53.77 }, { write: 15.90, read: 57.24 }, { write: 5.90, read: 20.15 }, { write: 15.21, read: 60.30 }, { write: 5.78, read: 18.26 }, { write: 12.50, read: 55.38 }, { write: 5.00, read: 17.76 }, { write: 2.11, read: 7.46 }, { write: 1.27, read: 2.32 }],
    FastLanes: [{ write: 1.11, read: 33.39 }, { write: 0.97, read: 18.67 }, { write: 1.02, read: 22.14 }, { write: 1.20, read: 55.61 }, { write: 0.65, read: 14.40 }, { write: 0.65, read: 14.59 }, { write: 0.65, read: 14.53 }, { write: 1.11, read: 22.57 }, { write: 0.98, read: 21.23 }, { write: 0.66, read: 16.30 }, { write: 0.76, read: 62.52 }, { write: 1.18, read: 51.57 }, { write: 1.08, read: 20.65 }, { write: 1.11, read: 26.10 }, { write: 1.16, read: 21.24 }, { write: 0.77, read: 24.43 }, { write: 1.15, read: 19.00 }, { write: 0.74, read: 23.29 }, { write: 1.07, read: 3.01 }, { write: 0.72, read: 3.04 }],
  },
};

const AGGREGATE: Record<Codec, AggRow[]> = {
  NONE: [
    { label: "PLAIN", gb: 32.366, bits: 50.65, write: 4.64, read: 10.57 },
    { label: "DELTA", gb: 8.225, bits: 12.87, write: 12.26, read: 25.74 },
    { label: "BSS", gb: 32.366, bits: 50.65, write: 4.71, read: 10.16 },
    { label: "FastLanes", gb: 9.045, bits: 14.15, write: 1.0, read: 26.61 },
    { label: "最优标准", gb: 8.188, bits: 12.81, write: 11.82, read: 26.14 },
    { label: "最优（含 FastLanes）", gb: 7.868, bits: 12.31, write: 1.46, read: 28.58, fastlanesColumns: 9 },
  ],
  SNAPPY: [
    { label: "PLAIN", gb: 16.254, bits: 25.43, write: 3.46, read: 15.02 },
    { label: "DELTA", gb: 7.673, bits: 12.01, write: 10.66, read: 25.95 },
    { label: "BSS", gb: 9.242, bits: 14.46, write: 7.18, read: 20.66 },
    { label: "FastLanes", gb: 8.5, bits: 13.3, write: 0.99, read: 27.42 },
    { label: "最优标准", gb: 7.644, bits: 11.96, write: 10.32, read: 26.4 },
    { label: "最优（含 FastLanes）", gb: 7.329, bits: 11.47, write: 1.47, read: 29.34, fastlanesColumns: 8 },
  ],
  ZSTD: [
    { label: "PLAIN", gb: 9.15, bits: 14.32, write: 2.73, read: 14.15 },
    { label: "DELTA", gb: 7.412, bits: 11.6, write: 6.1, read: 18.83 },
    { label: "BSS", gb: 7.253, bits: 11.35, write: 4.85, read: 18.96 },
    { label: "FastLanes", gb: 8.113, bits: 12.7, write: 0.92, read: 22.34 },
    { label: "最优标准", gb: 7.119, bits: 11.14, write: 5.39, read: 20.0 },
    { label: "最优（含 FastLanes）", gb: 6.98, bits: 10.92, write: 2.11, read: 22.45, fastlanesColumns: 5 },
  ],
};

const PLAN_TOTALS: Record<"SNAPPY" | "ZSTD", PlanTotal[]> = {
  SNAPPY: [
    { plan: "cudf-default", gb: 28.916, rewriteS: 59.2, readS: 5.82 },
    { plan: "best-standard", gb: 27.708, rewriteS: 56.3, readS: 5.62 },
    { plan: "best-with-fastlanes", gb: 27.393, rewriteS: 76.2, readS: 5.39 },
    { plan: "fastlanes-all", gb: 28.564, rewriteS: 87.8, readS: 5.53 },
  ],
  ZSTD: [
    { plan: "cudf-default", gb: 23.489, rewriteS: 61.9, readS: 5.35 },
    { plan: "best-standard", gb: 22.368, rewriteS: 59.6, readS: 5.18 },
    { plan: "best-with-fastlanes", gb: 22.228, rewriteS: 69.5, readS: 5.04 },
    { plan: "fastlanes-all", gb: 23.362, rewriteS: 92.3, readS: 4.91 },
  ],
};

const TABLE_ROWS: TableRow[] = [
  { table: "lineitem", codec: "SNAPPY", plans: [{ mb: 18277.9, rewriteS: 39.0, readS: 4.69 }, { mb: 17226.8, rewriteS: 37.4, readS: 4.5 }, { mb: 16959.8, rewriteS: 55.4, readS: 4.27 }, { mb: 17510.2, rewriteS: 63.3, readS: 4.43 }] },
  { table: "orders", codec: "SNAPPY", plans: [{ mb: 4994.3, rewriteS: 9.4, readS: 0.53 }, { mb: 4994.3, rewriteS: 8.7, readS: 0.55 }, { mb: 4955.9, rewriteS: 10.2, readS: 0.54 }, { mb: 5270.8, rewriteS: 12.3, readS: 0.52 }] },
  { table: "partsupp", codec: "SNAPPY", plans: [{ mb: 3983.0, rewriteS: 7.6, readS: 0.43 }, { mb: 3826.5, rewriteS: 7.0, readS: 0.41 }, { mb: 3816.3, rewriteS: 7.3, readS: 0.41 }, { mb: 4054.4, rewriteS: 8.6, readS: 0.41 }] },
  { table: "lineitem", codec: "ZSTD", plans: [{ mb: 15580.2, rewriteS: 42.9, readS: 3.48 }, { mb: 14571.6, rewriteS: 41.0, readS: 3.37 }, { mb: 14445.6, rewriteS: 48.8, readS: 3.24 }, { mb: 14994.0, rewriteS: 67.4, readS: 3.08 }] },
  { table: "orders", codec: "ZSTD", plans: [{ mb: 3796.8, rewriteS: 8.9, readS: 0.86 }, { mb: 3772.2, rewriteS: 8.7, readS: 0.82 }, { mb: 3765.6, rewriteS: 10.4, readS: 0.77 }, { mb: 4069.8, rewriteS: 12.8, readS: 0.8 }] },
  { table: "partsupp", codec: "ZSTD", plans: [{ mb: 2930.2, rewriteS: 7.0, readS: 0.74 }, { mb: 2843.4, rewriteS: 6.8, readS: 0.71 }, { mb: 2836.2, rewriteS: 7.2, readS: 0.75 }, { mb: 3050.0, rewriteS: 8.5, readS: 0.75 }] },
];

const ABLATION: AblationRow[] = [
  { column: "l_partkey", encoding: "FastLanes", rowsPerPage: 4999, bits: 25.38, write: 0.46, read: 17.86 },
  { column: "l_partkey", encoding: "DELTA", rowsPerPage: 4999, bits: 26.22, write: 8.28, read: 19.76 },
  { column: "l_partkey", encoding: "FastLanes", rowsPerPage: 19981, bits: 25.27, write: 1.07, read: 19.97 },
  { column: "l_partkey", encoding: "DELTA", rowsPerPage: 19981, bits: 26.05, write: 8.11, read: 20.03 },
  { column: "l_partkey", encoding: "FastLanes", rowsPerPage: 20480, bits: 25.07, write: 1.11, read: 21.27 },
  { column: "l_partkey", encoding: "DELTA", rowsPerPage: 20480, bits: 26.05, write: 8.24, read: 20.06 },
  { column: "l_shipdate", encoding: "FastLanes", rowsPerPage: 4999, bits: 12.29, write: 0.43, read: 15.28 },
  { column: "l_shipdate", encoding: "DELTA", rowsPerPage: 4999, bits: 12.57, write: 6.85, read: 15.34 },
  { column: "l_shipdate", encoding: "FastLanes", rowsPerPage: 19981, bits: 12.17, write: 0.66, read: 17.48 },
  { column: "l_shipdate", encoding: "DELTA", rowsPerPage: 19981, bits: 12.47, write: 6.92, read: 16.13 },
  { column: "l_shipdate", encoding: "FastLanes", rowsPerPage: 20480, bits: 12.07, write: 0.7, read: 20.09 },
  { column: "l_shipdate", encoding: "DELTA", rowsPerPage: 20480, bits: 12.47, write: 7.15, read: 16.68 },
];

const REASONS: { title: string; body: string }[] = [
  {
    title: "对未排序的值，FOR 比差分好",
    body: "FastLanes 按 page 存 value − page 最小值，每个 page 一个 bit 宽度。DELTA_BINARY_PACKED 存相邻值之差，数据没排序时差值范围是值范围的两倍，大约每值多 1 bit。l_partkey：25.07 对 26.05 bit。",
  },
  {
    title: "有序 key 正好相反",
    body: "相邻 key 只差 0 到几个单位，DELTA 只要 0.3-5 bit，剩下的大多还能被 codec 压掉；FastLanes 仍要覆盖整个 page 的范围，每值 13-17 bit。",
  },
  {
    title: "低基数列：不压缩时打平，压缩后吃亏",
    body: "l_linenumber、p_size 和 nation key 需要的 bit 数和字典索引一样。SNAPPY、ZSTD 还能继续压字典索引，却压不动 FastLanes 打包后的数据。常量列每值也要 1 bit，因为编码器从不选 0 bit。",
  },
  {
    title: "ZSTD 下日期更适合 BYTE_STREAM_SPLIT",
    body: "按字节拆开后，ZSTD 能对几乎不变的高位字节做熵编码，结果低于固定 bit 宽度：10.98-11.25 bit，FastLanes 是 11.91-11.93。",
  },
];

const PLAN_LABEL: Record<string, string> = {
  "cudf-default": "cuDF 默认",
  "best-standard": "最优标准",
  "best-with-fastlanes": "最优（含 FastLanes）",
  "fastlanes-all": "全部用 FastLanes",
};

function shortName(name: string): string {
  return name.split(".")[1];
}

function fmtPct(p: number): string {
  const abs = Math.abs(p);
  const body = abs >= 100 ? Math.round(abs).toLocaleString("en-US") : abs.toFixed(1);
  return `${p < 0 ? "−" : "+"}${body}%`;
}

function bestStandard(codec: Codec, i: number): { enc: Enc; bits: number } {
  let best: { enc: Enc; bits: number } = { enc: "PLAIN", bits: Number.POSITIVE_INFINITY };
  for (const enc of STANDARD) {
    const v = BITS[codec][enc][i];
    if (v !== null && v < best.bits) best = { enc, bits: v };
  }
  return best;
}

function CodecPicker({ value, onChange }: { value: Codec; onChange: (c: Codec) => void }) {
  return (
    <Row gap={6} align="center">
      <Text size="small" tone="secondary">
        压缩
      </Text>
      {CODECS.map((c) => (
        <span key={c}>
          <Pill active={c === value} onClick={() => onChange(c)}>
            {c}
          </Pill>
        </span>
      ))}
    </Row>
  );
}

export default function FastLanesTpchSf100Benchmark() {
  const theme = useHostTheme();
  const [codec, setCodec] = useState<Codec>("SNAPPY");

  const perColumn = COLUMNS.map((col, i) => {
    const best = bestStandard(codec, i);
    const fl = BITS[codec].FastLanes[i] ?? 0;
    const flSpeed = SPEED[codec].FastLanes[i];
    const stdSpeed = SPEED[codec][best.enc][i];
    return { col, best, fl, change: (100 * (fl - best.bits)) / best.bits, flSpeed, stdSpeed };
  });
  const wins = perColumn.filter((r) => r.change < 0).length;
  const agg = AGGREGATE[codec];
  const aggBest = agg.find((a) => a.label === "最优标准");
  const aggWithFl = agg.find((a) => a.label === "最优（含 FastLanes）");
  const savedPct = aggBest && aggWithFl ? (100 * (aggWithFl.gb - aggBest.gb)) / aggBest.gb : 0;

  const planRows = PLAN_TOTALS.SNAPPY.map((s, i) => {
    const z = PLAN_TOTALS.ZSTD[i];
    const sBase = PLAN_TOTALS.SNAPPY[1].gb;
    const zBase = PLAN_TOTALS.ZSTD[1].gb;
    return [
      PLAN_LABEL[s.plan],
      `${s.gb.toFixed(2)} GB`,
      i === 1 ? "-" : fmtPct((100 * (s.gb - sBase)) / sBase),
      `${z.gb.toFixed(2)} GB`,
      i === 1 ? "-" : fmtPct((100 * (z.gb - zBase)) / zBase),
      `${s.rewriteS.toFixed(0)} / ${z.rewriteS.toFixed(0)}`,
      `${s.readS.toFixed(2)} / ${z.readS.toFixed(2)}`,
    ];
  });

  return (
    <Stack gap={22}>
      <Stack gap={6}>
        <H1>FastLanes 与标准 Parquet 编码对比：TPC-H SF100</H1>
        <Text tone="secondary">
          cuDF 26.10 + FastLanes（<Code>fastlane-working</Code>），RTX 5090 Laptop GPU。8 张表（8.66 亿行）里所有适用列
          分别用 PLAIN、DICTIONARY、DELTA_BINARY_PACKED、BYTE_STREAM_SPLIT 和 FastLanes 编码，在 NONE、SNAPPY、ZSTD
          下各测一遍，page 为 20,480 行；然后按四种编码方案重写整表。
        </Text>
      </Stack>

      <Row gap={36} wrap>
        <Stat value="−1.1% / −0.6%" label="整个文件大小（SNAPPY / ZSTD），相对最优标准编码" tone="success" />
        <Stat value="9 / 8 / 5" label="23 个适用列中 FastLanes 最小的列数（NONE / SNAPPY / ZSTD）" />
        <Stat value="~1 GB/s" label="FastLanes 写入速度，DELTA 是 6-12 GB/s" tone="warning" />
        <Stat value="+3% ~ +19%" label="FastLanes 读取速度，相对 DELTA" tone="success" />
      </Row>

      <Callout tone="info" title="按列选用 FastLanes，不要整表都用">
        在未排序的 key、数量和日期列上，它比最优标准编码小 2-7%，读取也更快；在有序 key、低基数列和小表上要大得多，
        而且写入普遍慢 10 倍左右。适用列只占文件字节的 28-32%，所以整个文件只小 1% 左右。
      </Callout>

      <Divider />

      <Stack gap={12}>
        <Row justify="space-between" align="center" wrap gap={12}>
          <H2>逐列结果</H2>
          <CodecPicker value={codec} onChange={setCodec} />
        </Row>
        <Text tone="secondary">
          {codec} 下，FastLanes 在这 20 列中有 {wins} 列最小。每列都挑最小的编码（允许 FastLanes）时，这些列的总大小
          相对只在标准编码里挑的结果变化 {fmtPct(savedPct)}。
        </Text>
        <Grid columns="minmax(0, 1fr) minmax(0, 1fr)" gap={20} align="start">
          <Stack gap={6}>
            <H3>每列大小，{codec}（每值 bit）</H3>
            <BarChart
              horizontal
              height={680}
              categories={COLUMNS.map((c) => shortName(c.name))}
              series={[
                { name: "FastLanes", data: perColumn.map((r) => r.fl), tone: "info" },
                { name: "最优标准编码", data: perColumn.map((r) => r.best.bits), tone: "neutral" },
              ]}
              valueSuffix=" bit"
            />
            <Text size="small" tone="tertiary">
              每值 bit = column chunk 字节数 × 8 / 行数。20 个至少 100 万行的适用列。数据来源：fastlanes_encoding_bench
              逐列测试，2026-09-27。
            </Text>
          </Stack>
          <Stack gap={6}>
            <H3>FastLanes 对比最优标准编码，{codec}</H3>
            <Table
              headers={["列", "最优标准（bit）", "FastLanes（bit）", "大小变化", "读取 GB/s，FL / 标准"]}
              rows={perColumn.map((r) => [
                shortName(r.col.name),
                `${r.best.enc} ${r.best.bits.toFixed(2)}`,
                r.fl.toFixed(2),
                fmtPct(r.change),
                `${r.flSpeed.read.toFixed(1)} / ${r.stdSpeed.read.toFixed(1)}`,
              ])}
              columnAlign={["left", "left", "right", "right", "right"]}
              rowTone={perColumn.map((r) => (r.change < 0 ? "success" : undefined))}
              striped
            />
            <Text size="small" tone="tertiary">
              绿色行表示 FastLanes 更小。有 8 个高基数 key 请求 DICTIONARY 时 cuDF 实际写成了 DELTA_BINARY_PACKED，
              这些结果不参与最优标准编码的选择。
            </Text>
          </Stack>
        </Grid>
      </Stack>

      <Stack gap={12}>
        <H2>20 列汇总：大小和速度，{codec}</H2>
        <Grid columns={3} gap={20} align="start">
          <Stack gap={6}>
            <H3>总大小（GB）</H3>
            <BarChart
              categories={agg.map((a) => a.label)}
              series={[{ name: "大小", data: agg.map((a) => a.gb), tone: "info" }]}
              valueSuffix=" GB"
              showValues
              height={260}
              referenceLines={aggBest ? [{ value: aggBest.gb, label: "最优标准" }] : undefined}
            />
          </Stack>
          <Stack gap={6}>
            <H3>写入吞吐（GB/s）</H3>
            <BarChart
              categories={agg.map((a) => a.label)}
              series={[{ name: "写入", data: agg.map((a) => a.write), tone: "info" }]}
              valueSuffix=" GB/s"
              showValues
              height={260}
            />
          </Stack>
          <Stack gap={6}>
            <H3>读取吞吐（GB/s）</H3>
            <BarChart
              categories={agg.map((a) => a.label)}
              series={[{ name: "读取", data: agg.map((a) => a.read), tone: "info" }]}
              valueSuffix=" GB/s"
              showValues
              height={260}
            />
          </Stack>
        </Grid>
        <Text size="small" tone="tertiary">
          GB/s 按内存中的列数据算：字节数之和 / 中位时间之和（预热 1 次，计时 3 次）。写入 = 显存里的表写到 host buffer，
          读取 = host buffer 读回显存。"最优"两行按列挑最小的编码；{codec} 下有 {aggWithFl?.fastlanesColumns ?? 0} 列
          选中了 FastLanes。
        </Text>
      </Stack>

      <Divider />

      <Stack gap={12}>
        <H2>整表：8 张表，四种编码方案</H2>
        <Text tone="secondary">
          各方案只在 23 个适用列上不同，字符串和 decimal 列都用 cuDF 默认编码。用到 FastLanes 的文件都逐个 row group
          和源数据比对过。
        </Text>
        <Grid columns="minmax(0, 3fr) minmax(0, 2fr)" gap={20} align="start">
          <Table
            headers={["方案", "SNAPPY 大小", "相对最优标准", "ZSTD 大小", "相对最优标准", "重写 s（S / Z）", "热缓存读取 s（S / Z）"]}
            rows={planRows}
            columnAlign={["left", "right", "right", "right", "right", "right", "right"]}
            rowTone={[undefined, undefined, "success", undefined]}
          />
          <Stack gap={6}>
            <H3>8 张表的重写时间（s）</H3>
            <BarChart
              categories={PLAN_TOTALS.SNAPPY.map((p) => PLAN_LABEL[p.plan])}
              series={[
                { name: "SNAPPY", data: PLAN_TOTALS.SNAPPY.map((p) => p.rewriteS) },
                { name: "ZSTD", data: PLAN_TOTALS.ZSTD.map((p) => p.rewriteS) },
              ]}
              valueSuffix=" s"
              height={240}
            />
            <Text size="small" tone="tertiary">
              parquet_io_chunk 的处理时间：读源文件 + 编码 + 压缩 + 写入（不含校验）。
            </Text>
          </Stack>
        </Grid>
        <CollapsibleSection title="lineitem、orders、partsupp 的分表数据">
          <Table
            headers={["表", "Codec", "大小 MB：默认 / 最优标准 / 最优含 FL / 全部 FL", "重写 s", "热缓存读取 s"]}
            rows={TABLE_ROWS.map((t) => [
              t.table,
              t.codec,
              t.plans.map((p) => Math.round(p.mb).toLocaleString("en-US")).join(" / "),
              t.plans.map((p) => p.rewriteS.toFixed(1)).join(" / "),
              t.plans.map((p) => p.readS.toFixed(2)).join(" / "),
            ])}
            columnAlign={["left", "left", "right", "right", "right"]}
          />
          <Text size="small" tone="tertiary">
            part、customer、supplier、nation、region 没有收益：它们的适用列只有一个有序 key 和一个低基数列，最优方案
            就是标准编码。读取是 3 次的中位数，波动约 2%。
          </Text>
        </CollapsibleSection>
      </Stack>

      <Divider />

      <Stack gap={12}>
        <H2>FastLanes 为什么赢或输</H2>
        <Grid columns={2} gap={16}>
          {REASONS.map((r) => (
            <div
              key={r.title}
              style={{ padding: "10px 12px", borderLeft: `2px solid ${theme.stroke.secondary}` }}
            >
              <Text weight="semibold">{r.title}</Text>
              <Text size="small" tone="secondary">
                {r.body}
              </Text>
            </div>
          ))}
        </Grid>
      </Stack>

      <Stack gap={10}>
        <H3>Page 大小：FastLanes 有按 page 的固定开销（SNAPPY，lineitem）</H3>
        <Table
          headers={["列", "编码", "每页行数", "每值 bit", "写入 GB/s", "读取 GB/s"]}
          rows={ABLATION.map((a) => [
            a.column,
            a.encoding,
            a.rowsPerPage.toLocaleString("en-US"),
            a.bits.toFixed(2),
            a.write.toFixed(2),
            a.read.toFixed(2),
          ])}
          columnAlign={["left", "left", "right", "right", "right", "right"]}
          rowTone={ABLATION.map((a) => (a.encoding === "FastLanes" ? "info" : undefined))}
        />
        <Text size="small" tone="tertiary">
          5,000 行的 page 来自 614 行的 page 上限（cuDF 的 page 由完整的 5,000 行 fragment 组成）；19,981 行是 cuDF
          默认布局；20,480 行是其他测试统一用的对齐布局。各种布局下 FastLanes 都比 DELTA 小 2-4%，但 page 变大后它的
          写入速度提高一倍多。
        </Text>
      </Stack>

      <div style={{ height: 1, background: theme.stroke.tertiary }} />
      <Text size="small" tone="tertiary">
        注意：笔记本 GPU（取 3 次的中位数），host buffer 是 pageable 内存，整表读取走热缓存；sm_120 上 nvCOMP ZSTD
        配合 chunked reader 会卡住，所以读取改成按 row group 分批。完整报告：
        cpp/examples/parquet_io/docs/fastlanes/FASTLANES_TPCH_SF100_BENCHMARK_2026-09-27.md。原始数据：
        cpp/examples/parquet_io/artifacts/fastlanes_bench_sf100_20260927/。
      </Text>
    </Stack>
  );
}
