
print("Choose an Algorithm")
print("A. CPU Scheduling")
print("B. Memory Management (Fixed Partition)")
print("C. Disk Scheduling")


algo_sched = input("Choose a Letter [A-C]: ")


if algo_sched == "A":
    print("CPU Scheduling")
    print("Choose a type of CPU Scheduling")
    print("1. FCFS")
    print("2. SJF")
    print("3. SRT")
    print("4. PRIORITY")
    print("5. RR")
    print("6. HRRN")
   
    cpu_sched = int(input("Choose a number [1-6]: "))
    if cpu_sched == 1:
        class Process:
            def __init__(self, pid, arrival_time, burst_time):
                self.pid = pid
                self.arrival_time = arrival_time
                self.burst_time = burst_time
                self.completion_time = 0
                self.turnaround_time = 0
                self.waiting_time = 0
                self.response_time = 0


            def calculate_metrics(self, start_time):
                self.response_time = start_time - self.arrival_time if start_time >= self.arrival_time else 0
                self.completion_time = start_time + self.burst_time
                self.turnaround_time = self.completion_time - self.arrival_time
                self.waiting_time = self.turnaround_time - self.burst_time




        def fcfs_scheduling(processes):
            print("\nFCFS CPU Scheduling\n")
            processes.sort(key=lambda p: p.arrival_time)  # Sort by arrival time


            time = 0
            for process in processes:
                if time < process.arrival_time:
                    time = process.arrival_time
                process.calculate_metrics(time)
                time += process.burst_time


            # Print table headers with proper alignment
            print(f"{'Process':<10}{'Arrival Time':<15}{'Burst Time':<15}{'Completion Time':<20}{'Turnaround Time':<20}{'Waiting Time':<15}{'Response Time':<15}")
            for process in processes:
                print(f"P{process.pid:<9}{process.arrival_time:<15}{process.burst_time:<15}{process.completion_time:<20}{process.turnaround_time:<20}{process.waiting_time:<15}{process.response_time:<15}")


            avg_turnaround_time = sum(p.turnaround_time for p in processes) / len(processes)
            avg_waiting_time = sum(p.waiting_time for p in processes) / len(processes)
            avg_response_time = sum(p.response_time for p in processes) / len(processes)


            print(f"\nAverage Turnaround Time: {avg_turnaround_time:.2f}")
            print(f"Average Waiting Time: {avg_waiting_time:.2f}")
            print(f"Average Response Time: {avg_response_time:.2f}")




        n = int(input("Enter the number of processes: "))
        processes = []


        for i in range(n):
            arrival_time = int(input(f"Enter arrival time for process P{i+1}: "))
            burst_time = int(input(f"Enter burst time for process P{i+1}: "))
            processes.append(Process(i + 1, arrival_time, burst_time))


        fcfs_scheduling(processes)
                   
    elif cpu_sched == 2:
        # Function to find the waiting time for all processes
        def findWaitingTime(processes, n, bt, at, wt):
            wt[0] = 0  # Waiting time for the first process is always 0
   
            for i in range(1, n):
                wt[i] = bt[i-1] + wt[i-1]  # Waiting time is the sum of burst times before the current process


        # Function to find the turn around time for all processes
        def findTurnAroundTime(processes, n, bt, wt, tat):
            for i in range(n):
                tat[i] = bt[i] + wt[i]  # Turnaround time is burst time + waiting time


        # Function to find the response time for all processes
        def findResponseTime(wt, at, rt):
            for i in range(len(wt)):
                rt[i] = wt[i] - at[i]  # Response time is the waiting time minus arrival time


        # Function to calculate average time
        def findAvgTime(processes, n, bt, at):
            wt = [0] * n  # Initialize waiting time array
            tat = [0] * n  # Initialize turnaround time array
            rt = [0] * n   # Initialize response time array
   
            findWaitingTime(processes, n, bt, at, wt)  # Calculate waiting times
            findTurnAroundTime(processes, n, bt, wt, tat)  # Calculate turnaround times
            findResponseTime(wt, at, rt)  # Calculate response times
   
            total_wt = sum(wt)
            total_tat = sum(tat)
            total_rt = sum(rt)
   
            # Format and print the table with proper alignment
            print(f"{'Process':<10}{'Arrival Time':<15}{'Burst Time':<15}{'Waiting Time':<15}{'Turnaround Time':<20}{'Response Time':<15}")
            for i in range(n):
                print(f"{processes[i]:<10}{at[i]:<15}{bt[i]:<15}{wt[i]:<15}{tat[i]:<20}{rt[i]:<15}")
   
            print(f"\nAverage waiting time: {total_wt / n:.2f}")
            print(f"Average turnaround time: {total_tat / n:.2f}")
            print(f"Average response time: {total_rt / n:.2f}")


        # Function to implement SJF scheduling
        def sjfScheduling(processes, bt, at, n):
            # Sort the burst times and process IDs based on burst times
            sorted_processes = sorted(zip(processes, bt, at), key=lambda x: x[1])  # Sort by burst time
            sorted_processes = list(zip(*sorted_processes))  # Unzip into separate lists
            sorted_processes[0], sorted_processes[1], sorted_processes[2] = list(sorted_processes[0]), list(sorted_processes[1]), list(sorted_processes[2])


            # Calculate average waiting time, turnaround time, and response time
            findAvgTime(sorted_processes[0], n, sorted_processes[1], sorted_processes[2])


        # Driver code
        n = int(input("Enter the number of processes: "))
        processes = []
        burst_time = []
        arrival_time = []
   
        # Get user input for each process
        for i in range(n):
            process_id = int(input(f"Enter Process ID for process {i+1}: "))
            processes.append(process_id)
       
            bt = int(input(f"Enter Burst Time for process {process_id}: "))
            burst_time.append(bt)
       
            at = int(input(f"Enter Arrival Time for process {process_id}: "))
            arrival_time.append(at)


        # Run the SJF Scheduling algorithm
        sjfScheduling(processes, burst_time, arrival_time, n)
       
    elif cpu_sched == 3:
        import heapq


        # Function to find the waiting time for all processes
        def findWaitingTime(processes, n, bt, at, wt):
            wt[0] = 0  # Waiting time for the first process is always 0
   
            for i in range(1, n):
                wt[i] = bt[i-1] + wt[i-1]  # Waiting time is the sum of burst times before the current process


        # Function to find the turn around time for all processes
        def findTurnAroundTime(processes, n, bt, wt, tat):
            for i in range(n):
                tat[i] = bt[i] + wt[i]  # Turnaround time is burst time + waiting time


        # Function to find the response time for all processes
        def findResponseTime(wt, at, rt):
            for i in range(len(wt)):
                rt[i] = wt[i] - at[i]  # Response time is the waiting time minus arrival time


        # Function to calculate average time
        def findAvgTime(processes, n, bt, at):
            wt = [0] * n  # Initialize waiting time array
            tat = [0] * n  # Initialize turnaround time array
            rt = [0] * n   # Initialize response time array
   
            findWaitingTime(processes, n, bt, at, wt)  # Calculate waiting times
            findTurnAroundTime(processes, n, bt, wt, tat)  # Calculate turnaround times
            findResponseTime(wt, at, rt)  # Calculate response times
   
            total_wt = sum(wt)
            total_tat = sum(tat)
            total_rt = sum(rt)
   
            # Format and print the table with proper alignment
            print(f"{'Process':<10}{'Arrival Time':<15}{'Burst Time':<15}{'Waiting Time':<15}{'Turnaround Time':<20}{'Response Time':<15}")
            for i in range(n):
                print(f"{processes[i]:<10}{at[i]:<15}{bt[i]:<15}{wt[i]:<15}{tat[i]:<20}{rt[i]:<15}")
   
            print(f"\nAverage waiting time: {total_wt / n:.2f}")
            print(f"Average turnaround time: {total_tat / n:.2f}")
            print(f"Average response time: {total_rt / n:.2f}")


        # Function to implement SRT scheduling
        def srtScheduling(processes, bt, at, n):
            # Create a list to store remaining burst times for each process
            remaining_bt = bt[:]
   
            # Initialize variables
            current_time = 0
            completed = 0
            wt = [0] * n
            tat = [0] * n
            rt = [0] * n
            start_time = [-1] * n
   
            # Create a priority queue (min-heap) to manage processes based on remaining burst time
            pq = []
   
            # Add processes to the priority queue (min-heap) based on their arrival time
            while completed < n:
                # Add processes that have arrived at the current time to the priority queue
                for i in range(n):
                    if at[i] <= current_time and remaining_bt[i] > 0 and start_time[i] == -1:
                        heapq.heappush(pq, (remaining_bt[i], i))
                        start_time[i] = current_time  # Mark the start time for the process
               
                if pq:
                    # Pop the process with the smallest remaining burst time
                    bt_remaining, idx = heapq.heappop(pq)
           
                    # Process execution (decrease remaining burst time)
                    remaining_bt[idx] -= 1
                    current_time += 1
           
                    # If the process is completed
                    if remaining_bt[idx] == 0:
                        completed += 1
                        tat[idx] = current_time - at[idx]  # Turnaround time is current time - arrival time
                        wt[idx] = tat[idx] - bt[idx]  # Waiting time is turnaround time - burst time
                        rt[idx] = start_time[idx] - at[idx]  # Response time is first start time - arrival time
               
                    # If the process has remaining burst time, push it back to the queue
                    if remaining_bt[idx] > 0:
                        heapq.heappush(pq, (remaining_bt[idx], idx))
                else:
                    current_time += 1  # If no process is ready to execute, increment time
   
            # Calculate and print the results
            findAvgTime(processes, n, bt, at)


        # Driver code
        if __name__ == "__main__":
            # Get user input for number of processes
            n = int(input("Enter the number of processes: "))
   
            processes = []
            burst_time = []
            arrival_time = []
   
            # Get user input for each process
            for i in range(n):
                process_id = int(input(f"Enter Process ID for process {i+1}: "))
                processes.append(process_id)
       
                bt = int(input(f"Enter Burst Time for process {process_id}: "))
                burst_time.append(bt)
       
                at = int(input(f"Enter Arrival Time for process {process_id}: "))
                arrival_time.append(at)


            # Run the SRT Scheduling algorithm
            srtScheduling(processes, burst_time, arrival_time, n)


    elif cpu_sched == 4:
        # Function to calculate Completion Time, Turnaround Time, Waiting Time, and Response Time
        def priority_scheduling(arrival_time, burst_time, priority):
            n = len(arrival_time)
   
            # Initializing required lists
            completion_time = [0] * n
            turnaround_time = [0] * n
            waiting_time = [0] * n
            response_time = [0] * n
            remaining_burst_time = burst_time[:]
   
            # Creating a list of processes with arrival, burst time, and priority
            processes = []
            for i in range(n):
                processes.append([i, arrival_time[i], burst_time[i], priority[i]])
   
            # Sorting processes based on arrival time
            processes.sort(key=lambda x: x[1])


            # Start the first process at the arrival time
            current_time = 0
            completed_processes = 0
            while completed_processes < n:
                # Find the process with the highest priority that is ready to run
                ready_processes = [p for p in processes if p[1] <= current_time and remaining_burst_time[p[0]] > 0]
                if ready_processes:
                    # Sort by priority and pick the process with the highest priority (lowest priority number)
                    ready_processes.sort(key=lambda x: x[3])
                    process = ready_processes[0]
           
                    index = process[0]
                    remaining_burst_time[index] -= 1
                    current_time += 1
           
                    # Calculate response time if it's the first time the process is being executed
                    if remaining_burst_time[index] == burst_time[index] - 1:
                        response_time[index] = current_time - arrival_time[index]
               
                    # If the process finishes, update completion time and calculate turnaround and waiting time
                    if remaining_burst_time[index] == 0:
                        completion_time[index] = current_time
                        turnaround_time[index] = completion_time[index] - arrival_time[index]
                        waiting_time[index] = turnaround_time[index] - burst_time[index]
                        completed_processes += 1
                else:
                    current_time += 1
   
            return completion_time, turnaround_time, waiting_time, response_time


        # Function to calculate average times
        def calculate_averages(turnaround_time, waiting_time, response_time):
            n = len(turnaround_time)
            avg_turnaround_time = sum(turnaround_time) / n
            avg_waiting_time = sum(waiting_time) / n
            avg_response_time = sum(response_time) / n
            return avg_turnaround_time, avg_waiting_time, avg_response_time


        # User input for arrival time, burst time, and priority
        n = int(input("Enter the number of processes: "))


        arrival_time = []
        burst_time = []
        priority = []


        # Collecting input for each process
        for i in range(n):
            arrival = int(input(f"Enter arrival time for process {i}: "))
            burst = int(input(f"Enter burst time for process {i}: "))
            pri = int(input(f"Enter priority for process {i} (lower number indicates higher priority): "))
            arrival_time.append(arrival)
            burst_time.append(burst)
            priority.append(pri)


        # Run the priority scheduling algorithm
        completion_time, turnaround_time, waiting_time, response_time = priority_scheduling(arrival_time, burst_time, priority)


        # Calculate averages
        avg_turnaround_time, avg_waiting_time, avg_response_time = calculate_averages(turnaround_time, waiting_time, response_time)


        # Print the table with the results
        print("\nProcess | Arrival Time | Burst Time | Completion Time | Turnaround Time | Waiting Time | Response Time")
        for i in range(n):
            print(f"P{i}     | {arrival_time[i]}         | {burst_time[i]}         | {completion_time[i]}            | {turnaround_time[i]}            | {waiting_time[i]}          | {response_time[i]}")


        # Print average times
        print(f"\nAverage Turnaround Time: {avg_turnaround_time:.2f}")
        print(f"Average Waiting Time: {avg_waiting_time:.2f}")
        print(f"Average Response Time: {avg_response_time:.2f}")


    elif cpu_sched == 5:
        from collections import deque


        # Get number of processes
        n = int(input("Enter the number of processes: "))


        # Initialize an empty list for processes
        processes = []


        # Get arrival time and burst time for each process
        for i in range(n):
            pid = i + 1
            at = int(input(f"Enter arrival time for Process {pid}: "))
            bt = int(input(f"Enter burst time for Process {pid}: "))
            processes.append([pid, at, bt])


        # Get time quantum (time slice)
        time_quantum = int(input("Enter the time quantum: "))


        # Initialize required lists
        completion_time = [-1] * n
        turnaround_time = [0] * n
        waiting_time = [0] * n
        response_time = [-1] * n


        # Sort processes by arrival time
        processes.sort(key=lambda x: x[1])
        pid_to_index = {processes[i][0]: i for i in range(n)}  # Mapping PID to original index


        # Ready queue and remaining burst times
        queue = deque()
        remaining_burst = {p[0]: p[2] for p in processes}  # {PID: Remaining Burst Time}
        current_time = 0
        index = 0  # To track arrival of new processes


        # If first process arrives later, start from that time
        if processes[0][1] > 0:
            current_time = processes[0][1]


        # Enqueue first process
        while index < n and processes[index][1] <= current_time:
            queue.append(processes[index][0])
            index += 1


        # Gantt Chart Data
        gantt_chart = []
        time_chart = [current_time]


        # Process queue
        while queue:
            pid = queue.popleft()
            p_index = pid_to_index[pid]  # Get correct index after sorting


            # If first time running, set response time
            if response_time[p_index] == -1:
                response_time[p_index] = current_time - processes[p_index][1]


            # Execute process
            execution_time = min(time_quantum, remaining_burst[pid])
            current_time += execution_time
            remaining_burst[pid] -= execution_time


            # Gantt Chart Updates
            gantt_chart.append(f"P{pid}")
            time_chart.append(current_time)


            # Check for newly arrived processes
            while index < n and processes[index][1] <= current_time:
                queue.append(processes[index][0])
                index += 1


            # If process is not finished, re-add it to queue
            if remaining_burst[pid] > 0:
                queue.append(pid)
            else:
                completion_time[p_index] = current_time
                turnaround_time[p_index] = completion_time[p_index] - processes[p_index][1]
                waiting_time[p_index] = turnaround_time[p_index] - processes[p_index][2]


            # Handle case when queue is empty and there are still processes arriving
            if not queue and index < n:
                current_time = processes[index][1]  # Jump forward in time to next process arrival
                queue.append(processes[index][0])   # Enqueue the next process
                index += 1


        # Calculate averages
        avg_tat = sum(turnaround_time) / n
        avg_wt = sum(waiting_time) / n
        avg_rt = sum(response_time) / n


        # Display results
        print("\nPID  AT  BT  CT  TAT  WT  RT")
        for i in range(n):
            print(f"{processes[i][0]:3}  {processes[i][1]:2}  {processes[i][2]:2}  {completion_time[i]:2}  {turnaround_time[i]:3}  {waiting_time[i]:2}  {response_time[i]:2}")


        print(f"\nAverage Turnaround Time: {avg_tat:.2f}")
        print(f"Average Waiting Time: {avg_wt:.2f}")
        print(f"Average Response Time: {avg_rt:.2f}")


        # Display Gantt Chart
        print("\nGantt Chart:")
        print(" | ".join(gantt_chart))
        print(" ".join(str(t) for t in time_chart))
       
    elif cpu_sched == 6:
        # HRRN Scheduling Function
        def hrrn_scheduling(n, processes):
            # Sort processes by arrival time
            processes.sort(key=lambda x: x[1])


            # Initialize variables
            current_time = 0
            completed = 0
            gantt_chart = []
            completion_time = [-1] * n
            turnaround_time = [0] * n
            waiting_time = [0] * n
            response_time = [-1] * n
            executed_processes = []
            initial_rr_values = {}


            while completed < n:
                # Find processes that have arrived but are not yet completed
                available_processes = [p for p in processes if p[1] <= current_time and completion_time[p[0] - 1] == -1]


                if not available_processes:
                    current_time += 1  # Move time forward if no process is available
                    continue


                # Calculate response ratios
                response_ratios = []
                for p in available_processes:
                    pid, at, bt = p
                    waiting_time_now = current_time - at
                    response_ratio = (waiting_time_now + bt) / bt  # (WT + BT) / BT
                    response_ratios.append((response_ratio, p))
           
                    # Store initial response ratio before execution
                    if pid not in initial_rr_values:
                        initial_rr_values[pid] = response_ratio


                # Select process with highest response ratio
                selected_process = max(response_ratios, key=lambda x: x[0])[1]
                pid, at, bt = selected_process


                # Execute process
                if response_time[pid - 1] == -1:  # First execution
                    response_time[pid - 1] = current_time - at


                gantt_chart.append(f"P{pid}")
                executed_processes.append((pid, current_time))
                current_time += bt
                completion_time[pid - 1] = current_time
                turnaround_time[pid - 1] = completion_time[pid - 1] - at
                waiting_time[pid - 1] = turnaround_time[pid - 1] - bt


                completed += 1


            # Calculate averages
            avg_tat = sum(turnaround_time) / n
            avg_wt = sum(waiting_time) / n
            avg_rt = sum(response_time) / n


            # Display results
            print("\nPID  AT  BT  CT  TAT  WT  RT  Initial RR")
            for i in range(n):
                initial_rr = initial_rr_values[processes[i][0]]  # Fetch stored initial RR
                print(f"{processes[i][0]:3}  {processes[i][1]:2}  {processes[i][2]:2}  {completion_time[i]:2}  {turnaround_time[i]:3}  {waiting_time[i]:2}  {response_time[i]:2}  {initial_rr:.2f}")


            print(f"\nAverage Turnaround Time: {avg_tat:.2f}")
            print(f"Average Waiting Time: {avg_wt:.2f}")
            print(f"Average Response Time: {avg_rt:.2f}")


            # Display Gantt Chart
            print("\nGantt Chart:")
            print(" | ".join(gantt_chart))
            print(" ".join(str(t[1]) for t in executed_processes) + f" {current_time}")


        # Get input from user
        try:
            n = int(input("Enter number of processes: "))


            processes = []
            print("Enter Arrival Time and Burst Time for each process:")
            for i in range(n):
                at, bt = map(int, input(f"Process {i+1}: ").split())
                processes.append([i+1, at, bt])


            # Run HRRN scheduling
            hrrn_scheduling(n, processes)


        except Exception as e:
            print("Error:", e)


elif algo_sched == "B":
    print("Memory Management (Fixed Partition)")
    print("1. First Fit")
    print("2. Best Fit")
    print("3. Next Fit")
    print("4. Worst Fit")
   
    mem_sched = int(input("Choose a number [1-4]: "))
    if mem_sched == 1:
        def first_fit_memory_management(partitions, processes):
            allocation = [-1] * len(processes)  # Initialize allocation list (-1 means not allocated)


            # Traverse all processes and allocate memory
            for i in range(len(processes)):
                for j in range(len(partitions)):
                    if partitions[j] >= processes[i]:  # Check if partition is large enough for the process
                        allocation[i] = j  # Allocate the process to the partition
                        partitions[j] -= processes[i]  # Reduce the size of the partition
                        break  # Move to the next process after allocation
   
            # Print the result
            print(f"\n{'Process':<10}{'Memory Required':<20}{'Partition Allocated':<20}")
            for i in range(len(processes)):
                if allocation[i] != -1:
                    print(f"P{i + 1:<9}{processes[i]:<20}{'Partition ' + str(allocation[i] + 1)}")
                else:
                    print(f"P{i + 1:<9}{processes[i]:<20}{'Not Allocated'}")


        if __name__ == "__main__":
            # User inputs the partition sizes
            num_partitions = int(input("Enter the number of memory partitions: "))
            partitions = []
            for i in range(num_partitions):
                partition_size = int(input(f"Enter size of partition {i + 1}: "))
                partitions.append(partition_size)
   
            # User inputs the number of processes and their memory requirements
            num_processes = int(input("\nEnter the number of processes: "))
            processes = []
            for i in range(num_processes):
                process_memory = int(input(f"Enter memory required by process P{i + 1}: "))
                processes.append(process_memory)
   
        # Call the First Fit memory allocation function
        first_fit_memory_management(partitions, processes)
    elif mem_sched == 2:
        class BestFitMemoryManagement:
            def __init__(self, partition_sizes):
                self.partitions = partition_sizes  # List of partition sizes
                self.partitions_status = [None] * len(partition_sizes)  # None means empty partition


            def best_fit(self, process_size):
                # Find the best fit partition for the given process
                best_fit_index = -1
                min_diff = float('inf')


                for i in range(len(self.partitions)):
                    if self.partitions_status[i] is None and self.partitions[i] >= process_size:
                        diff = self.partitions[i] - process_size
                        if diff < min_diff:
                            min_diff = diff
                            best_fit_index = i


                if best_fit_index != -1:
                    self.partitions_status[best_fit_index] = process_size
                    print(f"Process of size {process_size} allocated to partition {best_fit_index} (Size {self.partitions[best_fit_index]})")
                else:
                    print(f"Process of size {process_size} could not be allocated.")


            def deallocate(self, partition_index):
                # Deallocate the partition
                if self.partitions_status[partition_index] is not None:
                    print(f"Partition {partition_index} deallocated (Process size: {self.partitions_status[partition_index]})")
                    self.partitions_status[partition_index] = None
                else:
                    print(f"Partition {partition_index} is already empty.")


            def display_partitions(self):
                print("\nCurrent Partition Status:")
                for i in range(len(self.partitions)):
                    if self.partitions_status[i] is None:
                        print(f"Partition {i}: Empty (Size {self.partitions[i]})")
                    else:
                        print(f"Partition {i}: Occupied (Process Size {self.partitions_status[i]})")




        # Input from the user
        def get_partition_sizes():
            partitions = []
            num_partitions = int(input("Enter the number of partitions: "))
            for i in range(num_partitions):
                size = int(input(f"Enter the size of partition {i + 1}: "))
                partitions.append(size)
            return partitions


        def get_process_sizes():
            processes = []
            num_processes = int(input("Enter the number of processes: "))
            for i in range(num_processes):
                size = int(input(f"Enter the size of process {i + 1}: "))
                processes.append(size)
            return processes


        # Main function
        def main():
            # Get partition and process sizes from user
            partition_sizes = get_partition_sizes()
            memory_manager = BestFitMemoryManagement(partition_sizes)


            # Allocate processes
            processes = get_process_sizes()
            for process in processes:
                memory_manager.best_fit(process)


            memory_manager.display_partitions()


            # Deallocate a partition if needed
            deallocate_choice = input("Do you want to deallocate any partition? (yes/no): ")
            if deallocate_choice.lower() == 'yes':
                partition_index = int(input("Enter the partition index to deallocate: "))
                memory_manager.deallocate(partition_index)


            memory_manager.display_partitions()


        if __name__ == "__main__":
            main()


    elif mem_sched == 3:
        def next_fit_fixed_partition(partition_sizes, job_sizes):
            allocation = [-1] * len(job_sizes)  # Stores partition index for each job (-1 means not allocated)
            last_allocated = 0  # Start from last allocated partition


            for i in range(len(job_sizes)):  # Iterate through each job
                allocated = False
                for j in range(len(partition_sizes)):  
                    index = (last_allocated + j) % len(partition_sizes)  # Circular search (Next Fit)
                    if partition_sizes[index] >= job_sizes[i]:  # Check if job fits in the partition
                        allocation[i] = index  # Allocate job to partition
                        partition_sizes[index] -= job_sizes[i]  # Reduce available space in partition
                        last_allocated = index  # Update last allocated partition
                        allocated = True
                        break  # Move to next job


            return allocation


        # Get user input for total memory and OS size
        total_memory = int(input("Enter total memory size: "))
        os_size = int(input("Enter OS size: "))


        # Calculate available memory
        available_memory = total_memory - os_size
        print(f"Available memory for partitions: {available_memory}\n")


        # Get user input for memory partitions
        num_partitions = int(input("Enter number of memory partitions: "))
        partition_sizes = list(map(int, input(f"Enter {num_partitions} partition sizes separated by space: ").split()))


        # Validate if partition sizes exceed available memory
        if sum(partition_sizes) > available_memory:
            print("\nError: Total partition size exceeds available memory!")
            exit()


        # Get user input for jobs
        num_jobs = int(input("\nEnter number of jobs: "))
        job_sizes = list(map(int, input(f"Enter {num_jobs} job sizes separated by space: ").split()))


        # Perform Next Fit Fixed Partition Allocation
        allocation = next_fit_fixed_partition(partition_sizes.copy(), job_sizes)


        # Display results in a table format
        print("\nJob No.  Job Size  Partition No.")
        for i in range(num_jobs):
            if allocation[i] != -1:
                print(f"{i+1:7}  {job_sizes[i]:8}  {allocation[i]+1:10}")  # Partition numbers start from 1
            else:
                print(f"{i+1:7}  {job_sizes[i]:8}  Not Allocated")


        # Text-based memory allocation chart
        print("\nMemory Allocation Chart:")
        for i in range(num_jobs):
            if allocation[i] != -1:
                print(f"Job {i+1} (Size {job_sizes[i]}) -> Partition {allocation[i]+1}")
            else:
                print(f"Job {i+1} (Size {job_sizes[i]}) -> Not Allocated")


    elif mem_sched == 4:
        class WorstFitMemoryManagement:
            def __init__(self, partition_sizes):
                self.partitions = partition_sizes  # List of partition sizes
                self.partitions_status = [None] * len(partition_sizes)  # None means empty partition


            def worst_fit(self, process_size):
                # Find the worst fit partition for the given process
                worst_fit_index = -1
                max_diff = -1


                for i in range(len(self.partitions)):
                    if self.partitions_status[i] is None and self.partitions[i] >= process_size:
                        diff = self.partitions[i] - process_size
                        if diff > max_diff:
                            max_diff = diff
                            worst_fit_index = i


                if worst_fit_index != -1:
                    self.partitions_status[worst_fit_index] = process_size
                    print(f"Process of size {process_size} allocated to partition {worst_fit_index} (Size {self.partitions[worst_fit_index]})")
                else:
                    print(f"Process of size {process_size} could not be allocated.")


            def deallocate(self, partition_index):
                # Deallocate the partition
                if self.partitions_status[partition_index] is not None:
                    print(f"Partition {partition_index} deallocated (Process size: {self.partitions_status[partition_index]})")
                    self.partitions_status[partition_index] = None
                else:
                    print(f"Partition {partition_index} is already empty.")


            def display_partitions(self):
                print("\nCurrent Partition Status:")
                for i in range(len(self.partitions)):
                    if self.partitions_status[i] is None:
                        print(f"Partition {i}: Empty (Size {self.partitions[i]})")
                    else:
                        print(f"Partition {i}: Occupied (Process Size {self.partitions_status[i]})")




        # Input from the user
        def get_partition_sizes():
            partitions = []
            num_partitions = int(input("Enter the number of partitions: "))
            for i in range(num_partitions):
                size = int(input(f"Enter the size of partition {i + 1}: "))
                partitions.append(size)
            return partitions


        def get_process_sizes():
            processes = []
            num_processes = int(input("Enter the number of processes: "))
            for i in range(num_processes):
                size = int(input(f"Enter the size of process {i + 1}: "))
                processes.append(size)
            return processes


        # Main function
        def main():
            # Get partition and process sizes from user
            partition_sizes = get_partition_sizes()
            memory_manager = WorstFitMemoryManagement(partition_sizes)


         # Allocate processes
            processes = get_process_sizes()
            for process in processes:
                memory_manager.worst_fit(process)


            memory_manager.display_partitions()


            # Deallocate a partition if needed
            deallocate_choice = input("Do you want to deallocate any partition? (yes/no): ")
            if deallocate_choice.lower() == 'yes':
                partition_index = int(input("Enter the partition index to deallocate: "))
                memory_manager.deallocate(partition_index)


            memory_manager.display_partitions()


        if __name__ == "__main__":
            main()


elif algo_sched == "C":
    print("Disk Scheduling")
    print("1. FCFS")
    print("2. SSTF")
    print("3. SCAN")
    print("4. C-SCAN")
    print("5. LOOK")
    print("6. C-LOOK")
   
    disk_sched = int(input("Choose a number[1-6]"))
   
    if disk_sched == 1:
        import matplotlib.pyplot as plt




        def fcfs(requests, head):
            """
            First-Come, First-Served (FCFS) disk scheduling algorithm.
   
            Parameters:
            requests (list): List of disk requests.
            head (int): Initial position of the disk head.
   
            Returns:
            tuple: A tuple containing the seek sequence and total seek count.
            """
            seek_sequence = []
            seek_count = 0
            current_position = head




            for request in requests:
                seek_sequence.append(request)
                seek_count += abs(current_position - request)  # Calculate seek time
                current_position = request  # Move head to the requested position




            return seek_sequence, seek_count
       
            # User input
        requests = list(map(int, input("Enter the disk requests (comma-separated): ").split(',')))
        head = int(input("Enter the initial head position: "))


        # Compute FCFS disk scheduling
        sequence, count = fcfs(requests, head)


        # Display results
        print(f"Seek Sequence: {sequence}")
        print(f"Total Seek Count: {count}")


        # Plotting the seek sequence
        positions = [head] + sequence  # Include the initial head position in the plot
        plt.plot(range(len(positions)), positions, marker='o', color='b', linestyle='-')
        plt.title('Disk Scheduling - FCFS Algorithm')
        plt.xlabel('Sequence Step')
        plt.ylabel('Disk Position')
        plt.grid(True)
        plt.show()


    elif disk_sched == 2:
        import matplotlib.pyplot as plt


        def sstf(requests, head):
            """
            Shortest Seek Time First (SSTF) disk scheduling algorithm.
   
            Parameters:
            requests (list): List of disk requests.
            head (int): Initial position of the disk head.
   
            Returns:
            tuple: A tuple containing the seek sequence and total seek count.
            """
            seek_sequence = []
            seek_count = 0
            current_position = head
            requests = requests.copy()  # Copy to avoid modifying the original list




            while requests:
                # Find the closest request to the current head position
                closest_request = min(requests, key=lambda x: abs(current_position - x))
                seek_sequence.append(closest_request)
                seek_count += abs(current_position - closest_request)  # Calculate seek time
                current_position = closest_request  # Move head to the closest request
                requests.remove(closest_request)  # Remove the processed request




            return seek_sequence, seek_count
       
        requests = list(map(int, input("Enter the disk requests (comma-separated): ").split(',')))
        head = int(input("Enter the initial head position: "))


        # Compute FCFS disk scheduling
        sequence, count = sstf(requests, head)


        # Display results
        print(f"Seek Sequence: {sequence}")
        print(f"Total Seek Count: {count}")


        # Plotting the seek sequence
        positions = [head] + sequence  # Include the initial head position in the plot
        plt.plot(range(len(positions)), positions, marker='o', color='b', linestyle='-')
        plt.title('Disk Scheduling - SSTF Algorithm')
        plt.xlabel('Sequence Step')
        plt.ylabel('Disk Position')
        plt.grid(True)
        plt.show()


    elif disk_sched == 3:
        import matplotlib.pyplot as plt


        def scan(requests, head, direction):
            """
            SCAN disk scheduling algorithm.
   
            Parameters:
            requests (list): List of disk requests.
            head (int): Initial position of the disk head.
            direction (str): Direction of head movement ('left' or 'right').
   
            Returns:
            tuple: A tuple containing the seek sequence and total seek count.
            """
            seek_sequence = []
            seek_count = 0
            current_position = head
            requests = sorted(requests)  # Sort requests for processing




            # Determine the order of processing based on the direction
            if direction == 'left':
                requests = [r for r in requests if r < head][::-1] + [0] + [r for r in requests if r >= head]
            else:
                requests = [r for r in requests if r >= head] + [199] + [r for r in requests if r < head][::-1]




            for request in requests:
                seek_sequence.append(request)
                seek_count += abs(current_position - request)  # Calculate seek time
                current_position = request  # Move head to the requested position




            return seek_sequence, seek_count
       
        requests = list(map(int, input("Enter the disk requests (comma-separated): ").split(',')))
        head = int(input("Enter the initial head position: "))
        direction = input("Enter direction (left/right): ").strip().lower()


        # Compute FCFS disk scheduling
        sequence, count = scan(requests, head, direction)


        # Display results
        print(f"Seek Sequence: {sequence}")
        print(f"Total Seek Count: {count}")


        # Plotting the seek sequence
        positions = [head] + sequence  # Include the initial head position in the plot
        plt.plot(range(len(positions)), positions, marker='o', color='b', linestyle='-')
        plt.title('Disk Scheduling - SCAN Algorithm')
        plt.xlabel('Sequence Step')
        plt.ylabel('Disk Position')
        plt.grid(True)
        plt.show()


    elif disk_sched == 4:
        import matplotlib.pyplot as plt


        def cscan(requests, head, direction):
            """
            Circular SCAN (C-SCAN) disk scheduling algorithm.
   
            Parameters:
            requests (list): List of disk requests.
            head (int): Initial position of the disk head.
            direction (str): Direction of head movement ('left' or 'right').
   
            Returns:
            tuple: A tuple containing the seek sequence and total seek count.
            """
            seek_sequence = []
            seek_count = 0
            current_position = head
            requests = sorted(requests)  # Sort requests for processing




            # Determine the order of processing based on the direction
            if direction == 'left':
                requests = [r for r in requests if r < head][::-1] + [0] + [r for r in requests if r >= head]
            else:
                requests = [r for r in requests if r >= head] + [199] + [r for r in requests if r < head]




            for request in requests:
                seek_sequence.append(request)
                seek_count += abs(current_position - request)  # Calculate seek time
                current_position = request  # Move head to the requested position




            return seek_sequence, seek_count
       
        requests = list(map(int, input("Enter the disk requests (comma-separated): ").split(',')))
        head = int(input("Enter the initial head position: "))
        direction = input("Enter direction (left/right): ").strip().lower()


        # Compute FCFS disk scheduling
        sequence, count = cscan(requests, head, direction)


        # Display results
        print(f"Seek Sequence: {sequence}")
        print(f"Total Seek Count: {count}")


        # Plotting the seek sequence
        positions = [head] + sequence  # Include the initial head position in the plot
        plt.plot(range(len(positions)), positions, marker='o', color='b', linestyle='-')
        plt.title('Disk Scheduling - CSCAN Algorithm')
        plt.xlabel('Sequence Step')
        plt.ylabel('Disk Position')
        plt.grid(True)
        plt.show()


    elif disk_sched == 5:
        import matplotlib.pyplot as plt


        def look(requests, head, direction):
            """
            LOOK disk scheduling algorithm.
   
            Parameters:
            requests (list): List of disk requests.
            head (int): Initial position of the disk head.
            direction (str): Direction of head movement ('left' or 'right').
   
            Returns:
            tuple: A tuple containing the seek sequence and total seek count.
            """
            seek_sequence = []
            seek_count = 0
            current_position = head
            requests = sorted(requests)  # Sort requests for processing




            # Determine the order of processing based on the direction
            if direction == 'left':
                requests = [r for r in requests if r < head][::-1] + [r for r in requests if r >= head]
            else:
                requests = [r for r in requests if r >= head] + [r for r in requests if r < head][::-1]




            for request in requests:
                seek_sequence.append(request)
                seek_count += abs(current_position - request)  # Calculate seek time
                current_position = request  # Move head to the requested position




            return seek_sequence, seek_count
       
        requests = list(map(int, input("Enter the disk requests (comma-separated): ").split(',')))
        head = int(input("Enter the initial head position: "))
        direction = input("Enter direction (left/right): ").strip().lower()


        # Compute FCFS disk scheduling
        sequence, count = look(requests, head, direction)


        # Display results
        print(f"Seek Sequence: {sequence}")
        print(f"Total Seek Count: {count}")


        # Plotting the seek sequence
        positions = [head] + sequence  # Include the initial head position in the plot
        plt.plot(range(len(positions)), positions, marker='o', color='b', linestyle='-')
        plt.title('Disk Scheduling - LOOK Algorithm')
        plt.xlabel('Sequence Step')
        plt.ylabel('Disk Position')
        plt.grid(True)
        plt.show()


    elif disk_sched == 6:
        import matplotlib.pyplot as plt


        def clook(requests, head, direction):
            """
            C-LOOK disk scheduling algorithm.
   
            Parameters:
            requests (list): List of disk requests.
            head (int): Initial position of the disk head.
            direction (str): Direction of head movement ('left' or 'right').
   
            Returns:
            tuple: A tuple containing the seek sequence and total seek count.
            """
            seek_sequence = []
            seek_count = 0
            current_position = head
            requests = sorted(requests)  # Sort requests for processing




            # Determine the order of processing based on the direction
            if direction == 'left':
                requests = [r for r in requests if r < head][::-1] + [r for r in requests if r >= head]
            else:
                requests = [r for r in requests if r >= head] + [r for r in requests if r < head]




            for request in requests:
                seek_sequence.append(request)
                seek_count += abs(current_position - request)  # Calculate seek time
                current_position = request  # Move head to the requested position




            return seek_sequence, seek_count


        requests = list(map(int, input("Enter the disk requests (comma-separated): ").split(',')))
        head = int(input("Enter the initial head position: "))
        direction = input("Enter direction (left/right): ").strip().lower()


        # Compute FCFS disk scheduling
        sequence, count = clook(requests, head, direction)


        # Display results
        print(f"Seek Sequence: {sequence}")
        print(f"Total Seek Count: {count}")


        # Plotting the seek sequence
        positions = [head] + sequence  # Include the initial head position in the plot
        plt.plot(range(len(positions)), positions, marker='o', color='b', linestyle='-')
        plt.title('Disk Scheduling - CLOOK Algorithm')
        plt.xlabel('Sequence Step')
        plt.ylabel('Disk Position')
        plt.grid(True)
        plt.show()

