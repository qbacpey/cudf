settings {
    nodaemon = true,
    statusInterval = 20,
    insist = true
}

-- Define a global exclusion list for all sync tasks.
-- This prevents syncing build artifacts, caches, logs, and local IDE/git files.
local_exclude = {
    '*.tmp', '*.bak', '*.log', '*.swp',
    -- Build outputs and caches
    'target/', 'build/', 'dist/',
    '__pycache__/', '.pytest_cache/', '.cache/',
    -- Rust specific
    'private-arrow-rs/',
    -- Local development and test data
    'test_data/', '.git/', '.idea/', '.vscode/',
    -- Specific file paths to exclude
    'parquet_manager/metadata/managed_parquet_index.csv',
    'parquet_manager/managed_parquet',
    'experiment_data'
}

-- Define all sync destinations in one place.
local destinations = {
    -- DGX01
    -- { host = "qchen@dgx01.lab.dm.informatik.tu-darmstadt.de", path = "/home/qchen/1126-SNAPPY-CASCADE-EXP/" },
    { host = "qchen@dgx01.lab.dm.informatik.tu-darmstadt.de", path = "/home/qchen/GPUFileFormat-cudf/" },
    -- FNG01
    { host = "qchen@fng01.lab.tuda.systems", path = "/home/qchen/GPUFileFormat-cudf/" },
    -- { host = "qchen@fng01.lab.tuda.systems", path = "/home/qchen/1126-SNAPPY-CASCADE-EXP/" },
    { host = "qchen@fng01.lab.tuda.systems", path = "/mnt/labstore/qchen/GPUFileFormat-cudf/" },
    -- { host = "qchen@fng01.lab.tuda.systems", path = "/home/qchen/1126-SNAPPY-CASCADE-EXP/" },
    -- FNG02
    -- { host = "qchen@fng02.lab.tuda.systems", path = "/home/qchen/ParquetRewriter/" },
    -- { host = "qchen@fng02.lab.tuda.systems", path = "/mnt/labstore/qchen/ParquetRewriter/" }
}

-- Loop through each destination and create a sync configuration.
for _, dest in ipairs(destinations) do
    sync {
        default.rsyncssh,
        source    = "./",
        host      = dest.host,
        targetdir = dest.path,
        delay     = 1,
        delete    = false,
        exclude   = local_exclude,
        ssh = {
            port = 22
        },
        rsync = {
            archive  = true,
            compress = true,
            verbose  = true
        }
    }
end