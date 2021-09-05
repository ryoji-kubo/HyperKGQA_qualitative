import os

'''
Program to run for qualitative analysis of fbwq. Groups the different inference paths to evaluate the accuracy for each curvature range
'''

inference_path = os.environ['INFERENCE_PATH']
qa_data_path = os.environ['DATA_PATH']


if __name__ == "__main__":
    readfile = '{}/fbwq_relation_performance.csv'.format(inference_path)
    outfile = '{}/fbwq_results_grouped.csv'.format(inference_path)

    big_list = []
    min_total_size = 50
    total_test_size = 0
    total_correct = 0.0
    lowest_curv = 0.0
    highest_curv = 0.0
    total_combined = 0

    ## file_name, dataset, kg_type, model, dim, ipath_list, lowest_curv, highest_curv, total_correct, total_test_size, total_ipath_combined, total_accuracy
    with open(readfile, 'r') as rfile:
        for line_num, small_line in enumerate(rfile):
            if line_num == 0:
                continue
            line_list = small_line.strip().split(',')
            file_name, dataset, kg_type, model, dim, _, ipath, curvature, _, correct, test_size, _ = line_list
            test_size = int(test_size)
            curvature = float(curvature)
            if test_size != 0:
                correct = float(correct)
            else:
                correct = 0.0
            ##  First line
            if line_num == 0:
                total_test_size = test_size
                total_correct = correct
                lowest_curv = curvature
                highest_curv = curvature
                if total_test_size != 0:
                    total_accuracy = total_correct/total_test_size
                else:
                    total_accuracy = 0.0
                total_combined = 1

                prev_file_name = file_name

                big_list.append([file_name, dataset, kg_type, model, dim, ipath, lowest_curv, highest_curv, total_correct, total_test_size, total_combined, total_accuracy])
                continue

            ##  Moved on to a new file
            if file_name != prev_file_name:

                total_test_size = test_size
                total_correct = correct
                lowest_curv = curvature
                highest_curv = curvature
                if total_test_size != 0:
                    total_accuracy = total_correct/total_test_size
                else:
                    total_accuracy = 0.0
                total_combined = 1

                prev_file_name = file_name
                
                big_list.append([file_name, dataset, kg_type, model, dim, ipath, lowest_curv, highest_curv, total_correct, total_test_size, total_combined, total_accuracy])
                continue

            ##  Not satisfied minimum total size
            if total_test_size < min_total_size:
                
                total_test_size += test_size
                big_list[-1][-3] = total_test_size

                total_correct += correct
                big_list[-1][-4] = total_correct

                big_list[-1][5] +=f'/{ipath}'

                if curvature < lowest_curv:
                    lowest_curv = curvature
                    big_list[-1][6] = lowest_curv
                if curvature > highest_curv:
                    highest_curv = curvature
                    big_list[-1][7] = highest_curv

                if total_test_size != 0:
                    total_accuracy = total_correct/total_test_size
                else:
                    total_accuracy = 0.0

                big_list[-1][-1] = total_accuracy

                total_combined += 1
                big_list[-1][-2] = total_combined
            

            ## Satisfied minimum total size
            else:

                total_test_size = test_size
                total_correct = correct
                lowest_curv = curvature
                highest_curv = curvature
                if total_test_size != 0:
                    total_accuracy = total_correct/total_test_size
                else:
                    total_accuracy = 0.0

                total_combined = 1

                prev_file_name = file_name
                
                big_list.append([file_name, dataset, kg_type, model, dim, ipath, lowest_curv, highest_curv, total_correct, total_test_size, total_combined, total_accuracy])


    with open(outfile, 'w') as ofile:
        ## file_name, dataset, kg_type, model, dim, ipath_list, lowest_curv, highest_curv, total_correct, total_test_size, total_ipath_combined, total_accuracy
        ofile.write('file name,dataset,kg type,model,dimension,inference path list,lowest curvature,highest curvature,total correct,total test size,total inference paths combined,total accuracy\n')
        for line in big_list:
            ofile.write(f'{line[0]},{line[1]},{line[2]},{line[3]},{line[4]},{line[5]},{line[6]},{line[7]},{line[8]},{line[9]},{line[10]},{line[11]}\n')

