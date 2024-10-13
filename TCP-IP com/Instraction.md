# BlueROV Navigation Control with ROS and MATLAB

This document contains methods to establish communication between a BlueROV robot (using ROS) and MATLAB using three different approaches:
1. **Using MATLAB's ROS Toolbox** 
2. **Using Raw TCP/IP Communication between ROS and MATLAB**
3. **Using ROS with Python via TCP/IP and MATLAB** (Recommended)

## Prerequisites
- **ROS (Robot Operating System)** installed on your robot or machine.
- **MATLAB** installed with **ROS Toolbox** (for Method 1).
- Basic understanding of ROS nodes, topics, and services.

---

## Method 1: MATLAB's ROS Toolbox

This is the easiest and most straightforward method to connect MATLAB and ROS, as MATLAB has built-in ROS support through the ROS Toolbox. Here's how to use it:

### Steps:

1. **Install ROS Toolbox in MATLAB**:
   If you haven't already, install the **ROS Toolbox** via the Add-Ons Explorer in MATLAB.

2. **Start the ROS Master**:
   Open a terminal and start `roscore`:
   ```bash
   roscore
   ```

3. **Connect MATLAB to ROS**:
   In MATLAB, initialize the ROS connection using `rosinit`:
   ```matlab
   rosinit('http://<ROS_MASTER_URI>:11311');  % Replace <ROS_MASTER_URI> with your ROS master IP
   ```

4. **Subscribe to ROS Topics**:
   You can subscribe to ROS topics and retrieve data from ROS:
   ```matlab
   sub = rossubscriber('/example_topic');
   data = receive(sub,10);  % Wait up to 10 seconds to receive a message
   disp(data);
   ```

5. **Publish Data to ROS**:
   MATLAB can publish data to ROS topics as well:
   ```matlab
   pub = rospublisher('/example_topic', 'std_msgs/Float64');
   msg = rosmessage(pub);
   msg.Data = 42.0;
   send(pub, msg);
   ```

6. **Disconnect from ROS**:
   When you're done, shut down the ROS connection:
   ```matlab
   rosshutdown;
   ```

---

## Method 2: Using Raw TCP/IP Communication

This method sets up a raw TCP/IP communication between ROS and MATLAB without using the ROS Toolbox.

### Steps:

### 1. Set up a TCP/IP Server in ROS (Python Example):
Create a simple TCP server in ROS using Python’s `socket` library.

```python
import socket

# Set up a TCP/IP server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 5000))  # Bind to IP address and port
server_socket.listen(1)

print('Waiting for connection...')
client_socket, addr = server_socket.accept()
print(f'Connection from {addr}')

while True:
    data = client_socket.recv(1024)
    if not data:
        break
    print('Received:', data.decode('utf-8'))
    # Send response back
    client_socket.sendall(b'Hello from ROS')

client_socket.close()
server_socket.close()
```

### 2. Set up a TCP/IP Client in MATLAB:
Connect MATLAB to the ROS TCP server.

```matlab
% Connect to the ROS TCP/IP server
t = tcpclient('192.168.1.10', 5000);  % Replace with your ROS machine IP address

% Send data from MATLAB to ROS
write(t, uint8('Hello from MATLAB'));

% Read data sent by ROS
data = read(t, t.BytesAvailable);
disp(char(data));  % Convert the byte data to character format for display
```

This setup allows continuous communication between MATLAB and ROS over TCP/IP.

---

## Method 3: Using ROS with Python via TCP/IP and MATLAB

This method uses ROS in Python to set up a ROS node that communicates via TCP/IP, and MATLAB communicates with this node using TCP.

### Steps:

### 1. ROS Python Script (TCP Server and ROS Publisher):

Create a ROS node that acts as a TCP server and publishes messages to ROS topics.

```python
import rospy
import socket
from std_msgs.msg import String

def tcp_server():
    rospy.init_node('tcp_server_node', anonymous=True)
    pub = rospy.Publisher('tcp_data', String, queue_size=10)

    # TCP server setup
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 5000))
    server_socket.listen(1)

    rospy.loginfo('Waiting for connection...')
    client_socket, addr = server_socket.accept()
    rospy.loginfo(f'Connection from {addr}')

    while not rospy.is_shutdown():
        data = client_socket.recv(1024)
        if not data:
            break
        rospy.loginfo(f'Received: {data.decode()}')
        pub.publish(data.decode())  # Publish data to ROS topic

    client_socket.close()
    server_socket.close()

if __name__ == '__main__':
    try:
        tcp_server()
    except rospy.ROSInterruptException:
        pass
```

### 2. MATLAB TCP Client:

Connect MATLAB to the ROS node via TCP/IP and communicate.

```matlab
% Connect to ROS via TCP/IP
t = tcpclient('192.168.1.10', 5000);  % Replace with your ROS machine IP address

% Send a message to the ROS node
write(t, uint8('Hello from MATLAB'));

% Read the response from ROS
data = read(t, t.BytesAvailable);
disp(char(data));  % Convert bytes to string and display
```

This method allows you to integrate Python-based ROS communication with MATLAB.

---

## Summary

In this repository, we have demonstrated three methods for interfacing **MATLAB** with **ROS** for controlling and monitoring a BlueROV robot:

- **Method 1:** Using MATLAB's **ROS Toolbox** for direct ROS communication.
- **Method 2:** Setting up raw **TCP/IP** communication between ROS and MATLAB.
- **Method 3:** Using **Python and TCP/IP** to create a communication bridge between ROS and MATLAB.

Feel free to choose the method that best suits your application.

---


