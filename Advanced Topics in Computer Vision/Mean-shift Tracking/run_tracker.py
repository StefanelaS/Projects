import time

import cv2
import matplotlib.pyplot as plt
from sequence_utils import VOTSequence
from ncc_tracker_example import NCCTracker, NCCParams
from ms_tracker import MeanShiftTracker, MSParams

#failed_frames_path = "C:/Users/Asus/Desktop/FAKS/ATCV/project 2/ms-material"

# set the path to directory where you have the sequences
dataset_path = 'vot2014' # TODO: set to the dataet path on your disk
sequences = ['bicycle', 'car', 'polarbear', 'hand2', 'sphere']  # choose the sequence you want to test
# visualization and setup parameters
win_name = 'Tracking window'
reinitialize = True
show_gt = False
video_delay = 15
font = cv2.FONT_HERSHEY_PLAIN

for sequence in sequences:
    # create sequence object
    sequence = VOTSequence(dataset_path, sequence)
    init_frame = 0
    n_failures = 0
    # create parameters and tracker objects
    #parameters = NCCParams()
    #tracker = NCCTracker(parameters)
    parameters = MSParams(0.5, 1, 16, 0.5)
    tracker = MeanShiftTracker(parameters)

    time_all = 0

    # initialize visualization window
    sequence.initialize_window(win_name)
    # tracking loop - goes over all frames in the video sequence
    frame_idx = 0
    while frame_idx < sequence.length():
        img = cv2.imread(sequence.frame(frame_idx))
        # initialize or track
        if frame_idx == init_frame:
            # initialize tracker (at the beginning of the sequence or after tracking failure)
            t_ = time.time()
            tracker.initialize(img, sequence.get_annotation(frame_idx, type='rectangle'))
            time_all += time.time() - t_
            predicted_bbox = sequence.get_annotation(frame_idx, type='rectangle')
        else:
            # track on current frame - predict bounding box
            t_ = time.time()
            predicted_bbox = tracker.track(img)
            time_all += time.time() - t_

        # calculate overlap (needed to determine failure of a tracker)
        gt_bb = sequence.get_annotation(frame_idx, type='rectangle')
        o = sequence.overlap(predicted_bbox, gt_bb)

        # draw ground-truth and predicted bounding boxes, frame numbers and show image
        if show_gt:
            sequence.draw_region(img, gt_bb, (0, 255, 0), 1)
            sequence.draw_region(img, predicted_bbox, (0, 0, 255), 2)
            sequence.draw_text(img, '%d/%d' % (frame_idx + 1, sequence.length()), (25, 25))
            sequence.draw_text(img, 'Fails: %d' % n_failures, (25, 55))
            sequence.show_image(img, video_delay)

        if o > 0 or not reinitialize:
            # increase frame counter by 1
            frame_idx += 1
        else:
            # save the current frame to a file
            #failure_filename = f"failure_{sequence}_{frame_idx}.png"
            #cv2.imwrite(failure_filename, img)
            # increase frame counter by 5 and set re-initialization to the next frame
            frame_idx += 5
            init_frame = frame_idx
            n_failures += 1

    print('Tracking speed: %.1f FPS' % (sequence.length() / time_all))
    print('Tracker failed %d times' % n_failures)
    print('Number of frames:', sequence.length())

#%% TRYING DIFFERENT ALPHAS
sequences = ['bicycle', 'car', 'polarbear', 'hand2', 'sphere'] 
alpha = [0.1, 0.2, 0.3, 0.4, 0.5]
all_failures = []
runtime = []

for i in range(0,len(sequences)):
    failures_seq = []
    for a in alpha:
        # set the path to directory where you have the sequences
        dataset_path = 'vot2014' # TODO: set to the dataet path on your disk
        sequence = sequences[i]
        # visualization and setup parameters
        win_name = 'Tracking window'
        reinitialize = True
        show_gt = False
        video_delay = 15
        font = cv2.FONT_HERSHEY_PLAIN
        # create sequence object
        sequence = VOTSequence(dataset_path, sequence)
        init_frame = 0
        n_failures = 0
        # create parameters and tracker objects
        #parameters = NCCParams()
        #tracker = NCCTracker(parameters)
        parameters = MSParams(0.5, 1, 16, a)
        tracker = MeanShiftTracker(parameters)

        time_all = 0

        # initialize visualization window
        sequence.initialize_window(win_name)
        # tracking loop - goes over all frames in the video sequence
        frame_idx = 0
        while frame_idx < sequence.length():
            img = cv2.imread(sequence.frame(frame_idx))
            # initialize or track
            if frame_idx == init_frame:
                # initialize tracker (at the beginning of the sequence or after tracking failure)
                t_ = time.time()
                tracker.initialize(img, sequence.get_annotation(frame_idx, type='rectangle'))
                time_all += time.time() - t_
                predicted_bbox = sequence.get_annotation(frame_idx, type='rectangle')
            else:
                # track on current frame - predict bounding box
                t_ = time.time()
                predicted_bbox = tracker.track(img)
                time_all += time.time() - t_

            # calculate overlap (needed to determine failure of a tracker)
            gt_bb = sequence.get_annotation(frame_idx, type='rectangle')
            o = sequence.overlap(predicted_bbox, gt_bb)

        # draw ground-truth and predicted bounding boxes, frame numbers and show image
            if show_gt:
                sequence.draw_region(img, gt_bb, (0, 255, 0), 1)
                sequence.draw_region(img, predicted_bbox, (0, 0, 255), 2)
                sequence.draw_text(img, '%d/%d' % (frame_idx + 1, sequence.length()), (25, 25))
                sequence.draw_text(img, 'Fails: %d' % n_failures, (25, 55))
                sequence.show_image(img, video_delay)

            if o > 0 or not reinitialize:
                # increase frame counter by 1
                frame_idx += 1
            else:
                # increase frame counter by 5 and set re-initialization to the next frame
                frame_idx += 5
                init_frame = frame_idx
                n_failures += 1
        failures_seq.append(n_failures)
    all_failures.append(failures_seq)
    #runtime.append(sequence.length() / time_all)
    #print('Tracking speed: %.1f FPS' % (sequence.length() / time_all))
    #print('Tracker failed %d times' % n_failures)
    #print('Number of frames:', sequence.length())
print(all_failures) 

plt.plot(alpha, all_failures[0], label = 'bicycle')
plt.plot(alpha, all_failures[1], label = 'car')
plt.plot(alpha, all_failures[2], label = 'polarbear')
plt.plot(alpha, all_failures[3], label = 'hand2')
plt.plot(alpha, all_failures[4], label = 'sphere')
plt.legend()
plt.xticks(alpha)
plt.xlabel('Alpha')
plt.ylabel('Number of failures')
plt.show()

#%% TRYING DIFFERENT NUMBER OF HISTOGRAM BINS

bins = [8, 16, 24, 32, 40, 48, 56, 64, 72]
sequences = ['bicycle', 'car', 'polarbear', 'hand2', 'sphere'] 
all_failures = []
runtime = []

for i in range(0,len(sequences)):
    failures_seq = []
    runtime_seq = []
    for b in bins:
        # set the path to directory where you have the sequences
        dataset_path = 'vot2014' # TODO: set to the dataet path on your disk
        sequence = sequences[i]
        # visualization and setup parameters
        win_name = 'Tracking window'
        reinitialize = True
        show_gt = False
        video_delay = 15
        font = cv2.FONT_HERSHEY_PLAIN
        # create sequence object
        sequence = VOTSequence(dataset_path, sequence)
        init_frame = 0
        n_failures = 0
        # create parameters and tracker objects
        #parameters = NCCParams()
        #tracker = NCCTracker(parameters)
        parameters = MSParams(0.5, 1, b, 0.02)
        tracker = MeanShiftTracker(parameters)

        time_all = 0

        # initialize visualization window
        sequence.initialize_window(win_name)
        # tracking loop - goes over all frames in the video sequence
        frame_idx = 0
        while frame_idx < sequence.length():
            img = cv2.imread(sequence.frame(frame_idx))
            # initialize or track
            if frame_idx == init_frame:
                # initialize tracker (at the beginning of the sequence or after tracking failure)
                t_ = time.time()
                tracker.initialize(img, sequence.get_annotation(frame_idx, type='rectangle'))
                time_all += time.time() - t_
                predicted_bbox = sequence.get_annotation(frame_idx, type='rectangle')
            else:
                # track on current frame - predict bounding box
                t_ = time.time()
                predicted_bbox = tracker.track(img)
                time_all += time.time() - t_

            # calculate overlap (needed to determine failure of a tracker)
            gt_bb = sequence.get_annotation(frame_idx, type='rectangle')
            o = sequence.overlap(predicted_bbox, gt_bb)

        # draw ground-truth and predicted bounding boxes, frame numbers and show image
            if show_gt:
                sequence.draw_region(img, gt_bb, (0, 255, 0), 1)
                sequence.draw_region(img, predicted_bbox, (0, 0, 255), 2)
                sequence.draw_text(img, '%d/%d' % (frame_idx + 1, sequence.length()), (25, 25))
                sequence.draw_text(img, 'Fails: %d' % n_failures, (25, 55))
                sequence.show_image(img, video_delay)

            if o > 0 or not reinitialize:
                # increase frame counter by 1
                frame_idx += 1
            else:
                # increase frame counter by 5 and set re-initialization to the next frame
                frame_idx += 5
                init_frame = frame_idx
                n_failures += 1
        failures_seq.append(n_failures)
        runtime_seq.append(sequence.length() / time_all)
    all_failures.append(failures_seq)
    runtime.append(runtime_seq)
    #runtime.append(sequence.length() / time_all)
    #print('Tracking speed: %.1f FPS' % (sequence.length() / time_all))
    #print('Tracker failed %d times' % n_failures)
    #print('Number of frames:', sequence.length())
print(all_failures) 

plt.plot(bins, all_failures[0], label = 'bicycle')
plt.plot(bins, all_failures[1], label = 'car')
plt.plot(bins, all_failures[2], label = 'polarbear')
plt.plot(bins, all_failures[3], label = 'hand2')
plt.plot(bins, all_failures[4], label = 'sphere')
plt.legend()
plt.xticks(bins)
plt.xlabel('Number of histogram bins')
plt.ylabel('Number of failures')
plt.show()

plt.plot(bins, runtime[0], label = 'bicycle', )
plt.plot(bins, runtime[1], label = 'car')
plt.plot(bins, runtime[2], label = 'polarbear')
plt.plot(bins, runtime[3], label = 'hand2')
plt.plot(bins, runtime[4], label = 'sphere')
plt.legend()
plt.xlabel('Number of histogram bins')
plt.ylabel('Frames per second')
plt.xticks(bins)
plt.show()

#%% TRYING DIFFERENT TOLERANCE VALUE

tolerance = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
sequences = ['bicycle', 'car', 'polarbear', 'hand2', 'sphere'] 
all_failures = []
runtime = []

for i in range(0,len(sequences)):
    failures_seq = []
    runtime_seq = []
    for t in tolerance:
        # set the path to directory where you have the sequences
        dataset_path = 'vot2014' # TODO: set to the dataet path on your disk
        sequence = sequences[i]
        # visualization and setup parameters
        win_name = 'Tracking window'
        reinitialize = True
        show_gt = False
        video_delay = 15
        font = cv2.FONT_HERSHEY_PLAIN
        # create sequence object
        sequence = VOTSequence(dataset_path, sequence)
        init_frame = 0
        n_failures = 0
        # create parameters and tracker objects
        #parameters = NCCParams()
        #tracker = NCCTracker(parameters)
        parameters = MSParams(0.5, t, 32, 0.02)
        tracker = MeanShiftTracker(parameters)

        time_all = 0

        # initialize visualization window
        sequence.initialize_window(win_name)
        # tracking loop - goes over all frames in the video sequence
        frame_idx = 0
        while frame_idx < sequence.length():
            img = cv2.imread(sequence.frame(frame_idx))
            # initialize or track
            if frame_idx == init_frame:
                # initialize tracker (at the beginning of the sequence or after tracking failure)
                t_ = time.time()
                tracker.initialize(img, sequence.get_annotation(frame_idx, type='rectangle'))
                time_all += time.time() - t_
                predicted_bbox = sequence.get_annotation(frame_idx, type='rectangle')
            else:
                # track on current frame - predict bounding box
                t_ = time.time()
                predicted_bbox = tracker.track(img)
                time_all += time.time() - t_

            # calculate overlap (needed to determine failure of a tracker)
            gt_bb = sequence.get_annotation(frame_idx, type='rectangle')
            o = sequence.overlap(predicted_bbox, gt_bb)

        # draw ground-truth and predicted bounding boxes, frame numbers and show image
            if show_gt:
                sequence.draw_region(img, gt_bb, (0, 255, 0), 1)
                sequence.draw_region(img, predicted_bbox, (0, 0, 255), 2)
                sequence.draw_text(img, '%d/%d' % (frame_idx + 1, sequence.length()), (25, 25))
                sequence.draw_text(img, 'Fails: %d' % n_failures, (25, 55))
                sequence.show_image(img, video_delay)

            if o > 0 or not reinitialize:
                # increase frame counter by 1
                frame_idx += 1
            else:
                # increase frame counter by 5 and set re-initialization to the next frame
                frame_idx += 5
                init_frame = frame_idx
                n_failures += 1
        failures_seq.append(n_failures)
        runtime_seq.append(sequence.length() / time_all)
    all_failures.append(failures_seq)
    runtime.append(runtime_seq)
    #runtime.append(sequence.length() / time_all)
    #print('Tracking speed: %.1f FPS' % (sequence.length() / time_all))
    #print('Tracker failed %d times' % n_failures)
    #print('Number of frames:', sequence.length())
print(all_failures) 

plt.plot(tolerance, all_failures[0], label = 'bicycle')
plt.plot(tolerance, all_failures[1], label = 'car')
plt.plot(tolerance, all_failures[2], label = 'polarbear')
plt.plot(tolerance, all_failures[3], label = 'hand2')
plt.plot(tolerance, all_failures[4], label = 'sphere')
plt.legend()
plt.xticks(tolerance)
plt.xlabel('Tolerance')
plt.ylabel('Number of failures')
plt.show()

plt.plot(tolerance, runtime[0], label = 'bicycle', )
plt.plot(tolerance, runtime[1], label = 'car')
plt.plot(tolerance, runtime[2], label = 'polarbear')
plt.plot(tolerance, runtime[3], label = 'hand2')
plt.plot(tolerance, runtime[4], label = 'sphere')
plt.legend()
plt.xlabel('Tolerance')
plt.ylabel('Frames per second')
plt.xticks(tolerance)
plt.show()