
import gc
import pandas as pd

import torch

from embed_structure_model import trans_basic_block, trans_basic_block_Config
from tm_vec_utils import featurize_prottrans, embed_tm_vec, encode, load_database, query

from transformers import T5EncoderModel, T5Tokenizer

from Bio import SeqIO
import time
from datetime import datetime
from pathlib import Path


#Load the ProtTrans model and ProtTrans tokenizer
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
gc.collect()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model = model.eval()


#TM-Vec model paths
#Use the "large" config so the encoder layer count matches the large checkpoint
tm_vec_model_cpnt = "/scratch/akeluska/tm-bench/tmvec_1_models/tm_vec_swiss_model_large.ckpt"
tm_vec_model_config = "/scratch/akeluska/tm-bench/tmvec_1_models/49181515_tm_vec_swiss_model_large_params.json"

#Load the TM-Vec model
tm_vec_model_config = trans_basic_block_Config.from_json(tm_vec_model_config)
model_deep = trans_basic_block.load_from_checkpoint(tm_vec_model_cpnt, config=tm_vec_model_config)
model_deep = model_deep.to(device)
model_deep = model_deep.eval()


sequence_path = '/scratch/akeluska/prot_distill_divide/data/fasta/cath-domain-seqs.fa'
record_ids = []
record_seqs = []
with open(sequence_path) as handle:
    for record in SeqIO.parse(handle, "fasta"):
        record_ids.append(record.id)
        record_seqs.append(str(record.seq))


#Let's encode 10K sequences ("queries") and search a lookup database with 1M sequences
start_time = time.time()

results_df = pd.DataFrame()
encode_df = pd.DataFrame()

for encoding_size in [10, 100, 1000, 10000, 50000]:
    encoding_seqs = record_seqs[0:encoding_size]
    encode_start = time.time()
    queries = encode(encoding_seqs, model_deep, model, tokenizer, device)
    encode_seconds = time.time() - encode_start

    print(encoding_size, encode_seconds)

    encode_df = pd.concat([encode_df, pd.DataFrame([{
            "encoding_size": encoding_size,
            "encode_seconds": encode_seconds,
        }])], ignore_index=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = Path("src/results/time_benchmarks") / f"tmvec1_{timestamp}"
output_dir.mkdir(parents=True, exist_ok=True)
encode_df.to_csv(output_dir / "encoding_times.csv", index=False)

for database_size in [1000, 10000, 100000]:
    for query_size in [10, 100, 1000]:
        query_seqs = record_seqs[0:query_size]
        encode_start = time.time()
        queries = encode(query_seqs, model_deep, model, tokenizer, device)
        encode_seconds = time.time() - encode_start

        #Now let's load a lookup database- here it consists of 500K sequences
        #Load the database that we will query
        #Make sure that the query database was encoded using the same model that's being applied to the query (i.e. CATH and CATH database)
        db_build_start = time.time()
        lookup_database = load_database("/scratch/akeluska/prot_distill_divide/data/cath_large.npy", database_size)
        db_build_seconds = time.time() - db_build_start

        #Search databases (FAISS similarity search only)
        search_start = time.time()
        k = 10 #Return 10 nearest neighbors for every query
        D, I = query(lookup_database, queries, k)
        search_seconds = time.time() - search_start

        total_seconds = encode_seconds + db_build_seconds + search_seconds

        print(
            f"Searching {query_size} queries vs {database_size} database sequences "
            f"(encode {encode_seconds:.3f}s, index build {db_build_seconds:.3f}s, search {search_seconds:.3f}s, total {total_seconds:.3f}s)"
        )

        results_df = pd.concat([results_df, pd.DataFrame([{
            "query_size": query_size,
            "database_size": database_size,
            "encode_seconds": encode_seconds,
            "db_build_seconds": db_build_seconds,
            "search_seconds": search_seconds,
            "total_seconds": total_seconds,
        }])], ignore_index=True)



results_df.to_csv(output_dir / "query_times.csv", index=False)