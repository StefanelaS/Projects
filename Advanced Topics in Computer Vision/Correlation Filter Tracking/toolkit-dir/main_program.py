import argparse
import os
import sys
import yaml
import os
import shutil

from utils.utils import load_tracker, load_dataset, trajectory_overlaps, count_failures, average_time
from utils.io_utils import read_regions, read_vector
from utils.export_utils import export_measures


def evaluate_tracker(workspace_path, tracker_id, enlarge_factor, gaussian_sigma, filter_lambda, update_factor):

    tracker_class = load_tracker(workspace_path, tracker_id)
    tracker = tracker_class(enlarge_factor, gaussian_sigma, filter_lambda, update_factor)

    dataset = load_dataset(workspace_path)

    results_dir = os.path.join(workspace_path, 'results', tracker.name())
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    tracker.evaluate(dataset, results_dir)
    print('Evaluation has been completed successfully.')



def tracking_analysis(workspace_path, tracker_id, enlarge_factor, gaussian_sigma, filter_lambda, update_factor):

    dataset = load_dataset(workspace_path)

    tracker_class = load_tracker(workspace_path, tracker_id)
    tracker = tracker_class(enlarge_factor, gaussian_sigma, filter_lambda, update_factor)

    print('Performing evaluation for tracker:', tracker.name())

    per_seq_overlaps = len(dataset.sequences) * [0]
    per_seq_failures = len(dataset.sequences) * [0]
    per_seq_time = len(dataset.sequences) * [0]

    for i, sequence in enumerate(dataset.sequences):
        
        results_path = os.path.join(workspace_path, 'results', tracker.name(), sequence.name, '%s_%03d.txt' % (sequence.name, 1))
        if not os.path.exists(results_path):
            print('Results does not exist (%s).' % results_path)
        
        time_path = os.path.join(workspace_path, 'results', tracker.name(), sequence.name, '%s_%03d_time.txt' % (sequence.name, 1))
        if not os.path.exists(time_path):
            print('Time file does not exist (%s).' % time_path)

        regions = read_regions(results_path)
        times = read_vector(time_path)

        overlaps, overlap_valid = trajectory_overlaps(regions, sequence.groundtruth)
        failures = count_failures(regions)
        t = average_time(times, regions)

        per_seq_overlaps[i] = sum(overlaps) / sum(overlap_valid)
        per_seq_failures[i] = failures
        per_seq_time[i] = t
    
    return export_measures(workspace_path, dataset, tracker, per_seq_overlaps, per_seq_failures, per_seq_time)

def main():
    parser = argparse.ArgumentParser(description='Tracker Evaluation Utility')

    parser.add_argument('--workspace_path', help='Path to the VOT workspace', required=True, action='store')
    parser.add_argument('--tracker', help='Tracker identifier', required=True, action='store')

    args = parser.parse_args()

    enlarge_factor = 1
    gaussian_sigma = 2
    update_factor = 0.1
    filter_lambda = 1

    enlarge_factors = [1, 1.5, 2, 3]
    gaussian_sigmas = [1, 2, 3, 4, 5]
    filter_lambda = 1
    update_factors = [0.1, 0.2, 0.3, 0.4, 0.5] 
    
    update_factors = [0.1] 
    
    
    results_corr_path = '../workspace-dir/results/Correlation'
    analysis_corr_path = '../workspace-dir/analysis/Correlation'


    for alpha in update_factors:  

        if os.path.exists(results_corr_path):
            shutil.rmtree(results_corr_path)
            print("Results Correlation directory removed")
        if os.path.exists(analysis_corr_path):
            shutil.rmtree(analysis_corr_path)
            print("Analysis Correlation directory removed")
        print("Update factor is:", alpha)
        evaluate_tracker(args.workspace_path, args.tracker, enlarge_factor, gaussian_sigma, filter_lambda, alpha)
        tracking_analysis(args.workspace_path, args.tracker, enlarge_factor, gaussian_sigma, filter_lambda, alpha)
        
    for sigma in gaussian_sigmas:  

        if os.path.exists(results_corr_path):
            shutil.rmtree(results_corr_path)
            print("Results Correlation directory removed")
        if os.path.exists(analysis_corr_path):
            shutil.rmtree(analysis_corr_path)
            print("Analysis Correlation directory removed")
        print("Gaussian sigma is:", sigma)
        evaluate_tracker(args.workspace_path, args.tracker, enlarge_factor, sigma, filter_lambda, update_factor)
        tracking_analysis(args.workspace_path, args.tracker, enlarge_factor, sigma, filter_lambda, update_factor)

    for ef in enlarge_factors:  

        if os.path.exists(results_corr_path):
            shutil.rmtree(results_corr_path)
            print("Results Correlation directory removed")
        if os.path.exists(analysis_corr_path):
            shutil.rmtree(analysis_corr_path)
            print("Analysis Correlation directory removed")
        print("Enlarge factor is:", ef)
        evaluate_tracker(args.workspace_path, args.tracker, ef, gaussian_sigma, filter_lambda, update_factor)
        tracking_analysis(args.workspace_path, args.tracker, ef, gaussian_sigma, filter_lambda, update_factor)
        
if __name__ == "__main__":
    main()

