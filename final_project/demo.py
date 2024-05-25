import numpy as np
from sklearn.cluster import KMeans
from pylsl import resolve_stream
from pylsl import StreamInlet
from time import monotonic, sleep
import queue
import time
import model
import serial
import winsound
import pandas as pd

kmeans_model = None
qsize = 30
save_name = 'subject1.csv'


def main():

    # ----------------- Test -----------------
    # test = thres[70:85].reshape(1, -1)
    # if kmeans_model is not None:
    #     new_data = np.array(test)
    #     predicted_label = kmeans_model.predict(new_data)
    #     print("Predicted Label:", predicted_label[0])
    # else:
    #     print("KMeans model is not defined.")
    # ----------------- Test -----------------

    q = queue.Queue(maxsize=qsize)
    streams = resolve_stream('name', 'OpenViBE Stream1')
    inlet = StreamInlet(streams[0])
    start = monotonic()
    timer_start = monotonic()
    Thres = []
    Time = []
    Slope = []
    collect_data = False
    # import argparse
    # parser = argparse.ArgumentParser(description="CECNL BCI 2023 Car Demo")
    # parser.add_argument("port_num", type=str, help="Arduino bluetooth serial port")
    # args = parser.parse_args()

    # ser = serial.Serial(args.port_num, 9600, timeout=1, write_timeout=1)

    while True:
        sample, timestamp = inlet.pull_chunk()

        if timestamp:
            sample = sample[0][0]
            while q.qsize() >= qsize:
               _ = q.get()
            q.put(sample)
            prev_time = 0 # time of the previous sample
            prev_ratio = 0 # ratio of the previous sample
            ratio = sum(list(q.queue)) / q.qsize()
            end = monotonic()
            
            time_ = end - start
           
            slope = (ratio - prev_ratio) / (time_ - prev_time)
            if not collect_data:
                Thres.append(ratio)
                Time.append(time_)
                Slope.append(slope)
                if end - timer_start > 9 and time_ < 25:
                    winsound.Beep(2500, 500)
                    timer_start = monotonic()
                
                
            print(f"time:  {time_} ratio: {ratio} slope: {slope}")
            if time_ > 30 and not collect_data: # 0~30s: collect data and calculate average threshold
                # Turn Thres, Slope, Time into a DF
                Thres = np.array(Thres)
                Slope = np.array(Slope)
                Time = np.array(Time)
                df = pd.DataFrame({'thres': Thres, 'slope': Slope, 'time': Time})
                kmeans_model = model.Kmeans_model(df, save_name, window = 15)
                hierarchical_model,label=model.Hierarchical_Clustering_model(df, save_name, window = 15)
                gmm=model.GaussianMixture_model(df, save_name, window = 15)
                # print("Average Threshold:", avg_thres)
                collect_data = True


            if collect_data:
                # get the input data
                input_data = pd.DataFrame([{'thres': ratio, 'slope': slope}])
                label = kmeans_model.predict(input_data)    # predict the label of the current sample
                print("Kmeans Label:", label)
                if label == 1 and q.qsize() == qsize:
                    print("move forward", ratio)
                    # ser.write(b'1')
                else:
                    print("stop ", ratio)
                    # ser.write(b'0')
                    
                hierarchical_model,label=model.Hierarchical_Clustering_model(df, save_name, window = 15)   
                print("Hierarchical Label:", label)
                if label == 1 and q.qsize() == qsize:
                    print("move forward", ratio)
                    # ser.write(b'1')
                else:
                    print("stop ", ratio)
                    # ser.write(b'0') 
                
                label=gmm.predict(input_data)
                print("Gausian Mixture Label:", label)
                if label == 1 and q.qsize() == qsize:
                    print("move forward", ratio)
                    # ser.write(b'1')
                else:
                    print("stop ", ratio)
                    # ser.write(b'0')

        time.sleep(0.2)




    # thres = np.array(thres)
    # time = np.array(time)
    # avg_thres = kmeans_f2(thres, time)
    # print(avg_thres)

    # while True:
    #     sample, timestamp = inlet.pull_chunk()
    #     if timestamp:
    #         sample = sample[0][0]
    #         # print(sample)  # find fish
    #         while q.qsize() >= qsize:
    #             _ = q.get()
    #         q.put(sample)

    #         ratio = sum(list(q.queue)) / q.qsize()

    #         if ratio > avg_thres and q.qsize() == qsize:
    #             print("move forward", ratio)
    #         else:
    #             print("stop ", ratio)
    #     time.sleep(0.2)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print()
        exit(0)




