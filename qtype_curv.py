import os
import numpy as np
from graph_curvature import *

parser = argparse.ArgumentParser(
    description="Computing the Curvatures for Different Question Types in MetaQA"
)
parser.add_argument(
    "--hops", type=str, default='2', help = "Number of edges to reach the answer"
)
parser.add_argument(
    "--dataset", type=str, default='MetaQA', help = "KG dataset (MetaQA, fbwq)"
)


qa_data_path = os.environ['DATA_PATH']
checkpoints = os.environ['CHECKPOINTS']
kge_path = os.environ['KGE_PATH']
config_files = os.environ['CONFIG_FILES']
inference_path = os.environ['INFERENCE_PATH']
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start


def compute_curvature_estimate_inference(samples, qtype2relation, entity_dict, outfile):
    with open(outfile,'w') as outfile:
        curvs = []
        # count = 0
        for qtype,relations_list in qtype2relation.items():   #for each qtype
            relations = []
            for relation in relations_list.split('\t'):
                relations.append(relation.split()[0])
            
            # print(samples[:,1].shape)
            mask = np.in1d(samples[:,1],relations)

            # if count == 0:
            #     for i in range(50):
            #         print("{}: {}\n".format(i+1, mask[i]))
            #     print(relations)
            r_samples = samples[mask, :]
            curv, n_nodes = compute_curvature(r_samples, entity_dict)
            curvs.append((curv, n_nodes))
            print(qtype, curv)
            outfile.write("{},{},{}\n".format(qtype, relations_list, curv))

            # count+=1
        # print(count)

def compute_curvature_estimate_inference_fbwq(samples, qtype, entity_dict, outfile):
     with open(outfile,'w') as outfile:
        count = 0

        for key,relations_list in qtype.items():   #for each qtype
            print('{}/{}'.format(count+1, len(qtype)))
            # print(samples[:,1].shape)
            mask = np.in1d(samples[:,1],relations_list)

            if count == 0:
                for i in range(20):
                    print("{}: {}\n".format(i+1, mask[i]))
                print(relations_list)
            r_samples = samples[mask, :]
            # print(r_samples[:20])
            # print(relations_list)
            curv, _ = compute_curvature(r_samples, entity_dict)
            print(key, curv)
            
            relations_string = ''
            for index, relation in enumerate(relations_list):
                if index == len(relations_list)-1:
                    relations_string += relation
                else:
                    relations_string += "{}\t".format(relation)

            outfile.write("{},{},{}\n".format(key, relations_string, curv))

            count+=1
        print(count)


if __name__ == '__main__':
    args = parser.parse_args()

    if args.dataset == 'MetaQA':
        metaqa_data_path = '{}/QA_data/MetaQA'.format(qa_data_path)
        test_qtype = '{}/qa_test_qtype_{}hop.txt'.format(metaqa_data_path, args.hops)

        outfile_full = '{}/qtype_curv_{}_full.txt'.format(inference_path, args.hops)
        outfile_half = '{}/qtype_curv_{}_half.txt'.format(inference_path, args.hops)

        ##  Read different Question Types
        qtypes = []
        with open(test_qtype, 'r') as file:
            for line in file.readlines():
                if line.strip() not in qtypes:
                    qtypes.append(line.strip())
                
        # print(qtypes, len(qtypes))

        ## Dictionary of Question Type to Relation for 1 hop
        qtype2relation_1 = {
            "actor_to_movie": "starred_actors (inv)",
            "director_to_movie": "directed_by (inv)",
            "movie_to_actor": "starred_actors",
            "movie_to_director": "directed_by",
            "movie_to_genre": "has_genre",
            "movie_to_language": "in_language",
            "movie_to_tags": "has_tags",
            "movie_to_writer": "written_by",
            "movie_to_year": "release_year",
            "tag_to_movie": "has_tags (inv)",
            "writer_to_movie": "written_by (inv)"
        }
        ##  Create a Dictionary of Question Type to Relations for 2, 3 hop
        qtype2relation_2 = {}

        if args.hops == '2':
            for qtype in qtypes:
                first_qtype = qtype[:find_nth(qtype, '_', 3)]
                second_qtype = qtype[find_nth(qtype,'_', 2)+1:]
                qtype2relation_2[qtype] = "{}\t{}".format(qtype2relation_1[first_qtype], qtype2relation_1[second_qtype])
            
            # for qtype, relations in qtype2relation_2.items():
            #     print("==={}:===\n".format(qtype))
            #     print("{}\n".format(relations))

        elif args.hops == '3':
            for qtype in qtypes:
                first_qtype = qtype[:find_nth(qtype, '_', 3)]
                second_qtype = qtype[find_nth(qtype,'_', 2)+1:find_nth(qtype, '_', 5)]
                third_qtype = qtype[find_nth(qtype,'_', 4)+1:]
                qtype2relation_2[qtype] = "{}\t{}\t{}".format(qtype2relation_1[first_qtype], qtype2relation_1[second_qtype], qtype2relation_1[third_qtype])
            # for qtype, relations in qtype2relation_2.items():
            #     print("==={}:===\n".format(qtype))
            #     print("{}\n".format(relations))

        ##  Compute the Curvatures for Different Question Types
        for kg_type in ['full', 'half']:
            dataset_path = "{}/data/MetaQA_{}".format(kge_path, kg_type)
            ## Reading KG dataset
            triplets = read_kg_triplets(dataset_path, relation = 'all', type = 'train')

            with open('{}/entity_ids.del'.format(dataset_path)) as f:
                entity_dict = {v: int(k) for line in f for (k, v) in [line.strip().split(None, 1)]}

            if kg_type == 'full':
                if args.hops == '1':
                    compute_curvature_estimate_inference(triplets, qtype2relation_1, entity_dict, outfile_full)
                else:
                    compute_curvature_estimate_inference(triplets, qtype2relation_2, entity_dict, outfile_full)
            else:
                if args.hops == '1':
                    compute_curvature_estimate_inference(triplets, qtype2relation_1, entity_dict, outfile_half)
                else:
                    compute_curvature_estimate_inference(triplets, qtype2relation_2, entity_dict, outfile_half)

    ##  for fbwq   
    else:
        fbwq_data_path = '{}/QA_data/WebQuestionsSP'.format(qa_data_path)
        pruning_file = '{}/pruning_test.txt'.format(fbwq_data_path)
        qtype = {}  ##  question type implies the different inferencial paths
        outfile_full = '{}/fbwq_curv_full.txt'.format(inference_path)
        outfile_half = '{}/fbwq_curv_half.txt'.format(inference_path)

        with open(pruning_file, 'r') as pruning_file:
            for line in pruning_file.readlines():
                plist = line.split('\t')
                if plist[1].strip() not in qtype.keys():
                    qtype[plist[1].split()] = [relation.strip() for relation in plist[1].split('|')]

        for kg_type in ['full', 'half']:
            dataset_path = "{}/data/fbwq_{}".format(kge_path, kg_type)
            ## Reading KG dataset
            triplets = read_kg_triplets(dataset_path, relation = 'all', type = 'train')
            with open('{}/entity_ids.del'.format(dataset_path)) as f:
                entity_dict = {v: int(k) for line in f for (k, v) in [line.strip().split(None, 1)]}
            
            if kg_type == 'full':
                compute_curvature_estimate_inference_fbwq(triplets, qtype, entity_dict, outfile_full)
            else:
                compute_curvature_estimate_inference_fbwq(triplets, qtype, entity_dict, outfile_half)


