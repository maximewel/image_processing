from algo import AlgorithmProcessor
from data_classes import HyperParametersBundle, MorphOperation, ResultBundle

from typing import List, Tuple
from dataclasses import dataclass
import sys
import os
import numpy as np
import cv2
from rich.progress import Progress, MofNCompleteColumn, TimeElapsedColumn, TextColumn, BarColumn

import matplotlib.pyplot as plt

import concurrent.futures
import pandas as pd

IMAGES_FILEPATH = "../dataset/png"

@dataclass
class HyperParametersBundleMatrix:
    """A single morph kernel is used throughought the project in order to reduce the
    number of kernel possibilities"""
    morph_kernel_options: List[Tuple] #Recommended: See cv2.getStructuringElement
    repair_lost_baterias_options: List[bool]

    dist_transform_morh_options: List[MorphOperation]
    binarized_dist_morph_options: List[MorphOperation]

    #Threshold to apply on the normalized distance image between [0.0, 1.0]
    dist_threshold_range_options: List[Tuple[float,float]]

    def size(self) -> int:
        """Return the product of the sum of all the options of this matrix"""
        return np.prod([len(option_list) for option_list in [
            self.morph_kernel_options,
            self.repair_lost_baterias_options,
            self.dist_transform_morh_options,
            self.binarized_dist_morph_options,
            self.dist_threshold_range_options,
        ]])

def load_image(image_name: str) -> np.ndarray:
    image_filepath = os.path.join(os.path.dirname(__file__) , f"{IMAGES_FILEPATH}/{image_name}")
    image = cv2.cvtColor(cv2.imread(image_filepath), cv2.COLOR_BGR2RGB)
    return image

def bundle_generator(matrix: HyperParametersBundleMatrix):
    """Generator object used to avoid having every bundle in memory"""
    for morph_kernel in matrix.morph_kernel_options:
        for repair in matrix.repair_lost_baterias_options:
                for dist_morph in matrix.dist_transform_morh_options:
                    for bin_morph in matrix.binarized_dist_morph_options:
                        for dist_treshold in matrix.dist_threshold_range_options:
                            yield HyperParametersBundle(morph_kernel, repair, dist_morph, bin_morph, dist_treshold)

def search_matrix(matrix: HyperParametersBundleMatrix):
    """Search every choice of the hyper parameter bundle matrix. Compare the differences and return the best one"""

    #Create generator that is able to create bundles and yield them when necessary, not wastime any memory
    hyper_parameter_bundles = bundle_generator(matrix)

    #As we have a relatively small dataset, take eveything in memory to reduce IO operations
    image_filepath = os.path.join(os.path.dirname(__file__), f"{IMAGES_FILEPATH}")
    image_names = os.listdir(image_filepath)
    image_name_bin = [(image_name, load_image(image_name)) for image_name in image_names]

    image_count = len(image_names)
    bundle_count = matrix.size()

    algorithm_processor = AlgorithmProcessor()
                
    print(f"Testing {bundle_count} hyper-parameter bundles on {image_count} images")

    min_diff = sys.maxsize
    best_hyper_parameters: HyperParametersBundle = None
    with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn()) as progress:
        bundle_progress = progress.add_task("[blue]Processing bundle...", total=bundle_count)
        image_progress = progress.add_task("[green]Processing image...", total=image_count)
        for bundle in hyper_parameter_bundles:
            progress.reset(image_progress)
            #Comopute on each image
            result_bundles = []
            for image_name, image in image_name_bin:
                res = algorithm_processor.process(image_name, image, bundle)
                result_bundles.append(res)
                progress.advance(image_progress)
            #Do total diff
            total_diff = sum([bundle.diff for bundle in result_bundles])
            #Compare to best
            if total_diff < min_diff:
                min_diff = total_diff
                best_hyper_parameters = bundle
            progress.advance(bundle_progress)

    print(f"Finished, best hyp bundle: diff={min_diff} with hyper-parameters \n{best_hyper_parameters}")

def search_matrix_parallel(matrix: HyperParametersBundleMatrix, nb_proc: int = 1) -> None:
    """Search every choice of the hyper parameter bundle matrix. Compare the differences and return the best one"""
    if nb_proc <= 0:
        nb_proc = 1

    if nb_proc == 1:
        return search_matrix(matrix)

    #Generate every bundle
    hyper_parameter_bundles = [yield_bundle for yield_bundle in bundle_generator(matrix)]

    #As we have a relatively small dataset, take eveything in memory to reduce IO operations
    image_filepath = os.path.join(os.path.dirname(__file__), f"{IMAGES_FILEPATH}")
    image_names = os.listdir(image_filepath)
    image_name_bin = [(image_name, load_image(image_name)) for image_name in image_names]

    image_count = len(image_names)
    bundle_count = matrix.size()

    algorithm_processor = AlgorithmProcessor()
                
    print(f"Testing {bundle_count} hyper-parameter bundles on {image_count} images on {nb_proc} threads")

    #Cut bundles into list
    bundle_list_size = int(bundle_count/nb_proc)
    bundle_lists = []
    for i in range(0, nb_proc):
        start, end = i*bundle_list_size, (i+1)*bundle_list_size
        if end > bundle_count:
            end = bundle_count
        bundle_lists.append(hyper_parameter_bundles[start:end])
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=nb_proc) as executor:
        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), refresh_per_second=1) as progress:
            futures = []
            for i, bundle_list in enumerate(bundle_lists):
                bundle_progress = progress.add_task(f"[blue]Thread {i} - Processing bundle...", total=len(bundle_list))
                futures.append(executor.submit(execute_bundle_list, algorithm_processor, bundle_list, image_name_bin, progress, bundle_progress))

            result_bundles: List[Tuple[HyperParametersBundle, List [ResultBundle]]] = []
            for future in futures:
                result_bundles.extend(future.result())

        print(f"Computation finished")
    
    print(f"Preliminary analysis")
    analyze_results(result_bundles)

    print(f"Saving as dataframe")
    save_as_df(result_bundles)

def save_as_df(bundle_list: List[Tuple[HyperParametersBundle, List[ResultBundle]]]):
    """Save the list of bundle as a dataframe to perform data analysis later"""
    cols = ["morph_kernel", "repair_lost_bact", "dist_transf_morph", "bin_morph", "dist_threshold", "diffes", "diff", "times", "total_time"]
    data = []

    for hyper_parameter_bundle, result_bundles in bundle_list:
        diffes = [result_bundle.diff for result_bundle in result_bundles]
        times = [result_bundle.runtime_s for result_bundle in result_bundles]
        data.append([hyper_parameter_bundle.morph_kernel, hyper_parameter_bundle.repair_lost_baterias,
            hyper_parameter_bundle.dist_transform_morh, hyper_parameter_bundle.binarized_dist_morph,
            hyper_parameter_bundle.dist_threshold_range,
            diffes, sum(diffes),
            times, sum(times)])

    df = pd.DataFrame(data, columns=cols)
    filename = os.path.join(os.path.dirname(__file__) , "../out/dataframe.pkl")
    df.to_pickle(filename)
    print(f"Results saved as DF saved to {filename}")

def analyze_results(bundle_list: List[Tuple[HyperParametersBundle, List[ResultBundle]]]):
    """Do a light analysis of the results by displaying the best/worst/avg, displaying all diffes computed, and saving the best bundles' report."""
    diffes = np.array([sum([result.diff for result in results]) for _, results in bundle_list], dtype=np.int32)

    min_diff = np.min(diffes)
    max_diff = np.max(diffes)
    average_diff = np.average(diffes)

    best_bundles = np.array(bundle_list, dtype=object)[np.where(diffes == min_diff)[0]]
    worst_bundles = np.array(bundle_list, dtype=object)[np.where(diffes == max_diff)[0]]

    print(f"Got {len(bundle_list)} bundles, average diff: {average_diff}, max: {max_diff} ({len(worst_bundles)} bundles), min: {min_diff} ({len(best_bundles)} bundles)")

    fig, axes = plt.subplots(1, 2)
    x = np.arange(len(diffes))
    axes[0].plot(x, diffes)
    axes[0].set_xlabel('Bundle index')
    axes[0].set_ylabel('Bundle difference score')
    axes[0].set_title('Bundle differences')

    bp = axes[1].boxplot(diffes, vert = False)
    axes[1].set_xlabel("Difference")
    axes[1].set_xlim((-10, max(diffes)+10))
    axes[1].set_xticks(np.arange(-10, max(diffes)+10, 10))
    axes[1].yaxis.set_visible(False)
    axes[1].set_title("Whisper distribution")

    ## Style From https://machinelearningknowledge.ai/matplotlib-boxplot-tutorial-for-beginners/
    # changing color and linewidth of whiskers 
    for whisker in bp['whiskers']: whisker.set(color ='#8B008B', linewidth = 1.5, linestyle =":") 
    # changing color and linewidth of caps 
    for cap in bp['caps']: cap.set(color ='#8B008B', linewidth = 2) 
    # changing color and linewidth of medians 
    for median in bp['medians']: median.set(color ='red', linewidth = 3) 
    # changing style of fliers 
    for flier in bp['fliers']: flier.set(marker ='D', color ='#e7298a', alpha = 0.5) 

    fig.tight_layout()
    plt.show()
    plt.close(fig)

    dir_path = os.path.join(os.path.dirname(__file__) , f"../out/best_bundles")
    print(f"Best bundles:")
    for i, (hyp, results) in enumerate(sorted(best_bundles, key = lambda bundle: bundle_total_time(bundle[1]))):
        print(f"\nBundle {i}\n{hyp}")
        dir_name = f"best_bundle_{i}"
        os.makedirs(f"{dir_path}/{dir_name}")
        for res in results:
            res: ResultBundle
            fig = res.figure_all()
            plt.savefig(f'{dir_path}/{dir_name}/{res.image_name}_processing.png')
            plt.close(fig)

            fig = res.summary()
            plt.savefig(f'{dir_path}/{dir_name}/{res.image_name}_summary.png')
            plt.close(fig)

            cv2.imwrite(f'{dir_path}/{dir_name}/{res.image_name}_result.png', res.annotated_image)
        
        with open(f'{dir_path}/{dir_name}/bundle.txt', "w") as f:
            runtimes = [result.runtime_s for result in results]
            diffes = [result.diff for result in results]
            f.write(f"Hyper parameters:\n\n{hyp}\n\nResults: \n\tDifferences={sum(diffes)}({diffes})\n\tRuntime={sum(runtimes)}({runtimes})")

def bundle_total_time(results: List[ResultBundle]) -> float:
    times = [result.runtime_s for result in results]
    return sum(times)

def execute_bundle_list(algorithm_processor: AlgorithmProcessor, bundle_list: List[HyperParametersBundle], image_name_bin: Tuple[str, np.ndarray], progress: Progress, task_id: int):
    thread_bundles = []
    for bundle in bundle_list:
        #Comopute on each image
        result_bundles = []
        for image_name, image in image_name_bin:
            res = algorithm_processor.process(image_name, image, bundle)
            result_bundles.append(res)
        
        thread_bundles.append((bundle, result_bundles))
        progress.advance(task_id)
        progress.refresh()

    return thread_bundles