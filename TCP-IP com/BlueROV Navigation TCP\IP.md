# BlueROV Navigation with ROS, Gazebo, Unity, and MATLAB via TCP/IP

This document provides a guide on how to connect and control the **BlueROV** simulation in **Gazebo**, visualize it in **Unity**, and interface with **ROS** and **MATLAB** using **TCP/IP communication**. Specifically, this guide focuses on using ROS with Python and MATLAB to enable communication between all systems.

## Prerequisites

To set up this environment, ensure you have the following installed:

- **ROS (Robot Operating System)**, preferably ROS Noetic (on Ubuntu 20.04)
- **Gazebo** simulator (installed alongside ROS)
- **Unity** (for visualization) with **ROS#** package
- **MATLAB** with basic TCP/IP socket support (or ROS Toolbox if preferred)
- **Python 3.x** (to handle TCP/IP communication within ROS)

---

## Project Overview

This setup includes:

1. **Gazebo** to simulate the BlueROV robot.
2. **ROS** to control the BlueROV and interact with the Gazebo simulator.
3. **Unity** to provide real-time 3D visualization of the robot's simulation.
4. **MATLAB** to control and receive feedback from the robot using a custom TCP/IP connection.

### Communication Flow:

1. **Gazebo** simulates the physical model of the BlueROV, controlled by ROS.
2. **MATLAB** runs control algorithms and communicates with ROS through a Python-based TCP/IP server.
3. **Unity** visualizes the BlueROV and interacts with ROS via **ROS#** or TCP/IP communication.
4. **ROS** bridges the different systems, handling data from sensors and actuators while ensuring all systems communicate efficiently.

---

## Step 1: Set Up Gazebo with BlueROV and ROS

Gazebo will simulate the BlueROV underwater robot with ROS as the middleware for controlling the robot and receiving sensor data.

### 1. Install ROS and Gazebo:

If you haven't installed ROS and Gazebo yet, follow the ROS installation guide for your version:
- [ROS Noetic Installation Guide](http://wiki.ros.org/noetic/Installation/Ubuntu)

After installing ROS, install the necessary Gazebo plugins:
```bash
sudo apt install ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control
```

### 2. Set Up the BlueROV Model in Gazebo:

- Download or use an existing BlueROV ROS package. If there’s a pre-configured BlueROV model, load it in Gazebo.
- Ensure the BlueROV’s URDF model is set up to publish sensor data (IMU, camera, etc.) and accept control commands (velocity, thrust, etc.).

Start the simulation using a ROS launch file for BlueROV in Gazebo:
```bash
roslaunch bluerov_gazebo bluerov_simulation.launch
```

This will start the BlueROV in the Gazebo simulation environment.

---

## Step 2: Set Up Unity with ROS Integration

Unity can be used to provide high-quality rendering and visualization of the robot’s environment.

### Option 1: Using ROS# to Integrate ROS and Unity

1. **Install ROS# in Unity**:
   - Download ROS# from [here](https://github.com/siemens/ros-sharp) and import it into your Unity project.
   - Configure ROS# to connect Unity with the ROS system.

2. **Create Unity Scenes**:
   - Build a Unity scene that visualizes the BlueROV's movement, synchronized with the simulation running in Gazebo.
   - Use ROS# to subscribe to ROS topics such as `/camera` or `/rov_pose`, and publish control commands to ROS.

### Option 2: TCP/IP Communication between Unity and ROS

Alternatively, set up custom TCP/IP communication between Unity and ROS using sockets. Unity can act as a TCP/IP client or server, depending on your setup.

---

## Step 3: Python-based ROS Node for TCP/IP Communication

Here, we’ll use a Python-based ROS node that sets up a TCP/IP server. MATLAB can connect to this server to exchange data with ROS.

### 1. Create a TCP/IP Server in ROS (Python):

This server will receive commands from MATLAB and publish them as ROS topics to control the BlueROV, while also sending feedback from ROS back to MATLAB.

```python
import rospy
import socket
from std_msgs.msg import String, Float64

def tcp_server():
    rospy.init_node('tcp_server_node', anonymous=True)

    # Create ROS publishers for the BlueROV control
    control_pub = rospy.Publisher('/rov/control', String, queue_size=10)
    sensor_pub = rospy.Publisher('/rov/sensor_data', Float64, queue_size=10)

    # Set up the TCP/IP server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 5000))
    server_socket.listen(1)

    rospy.loginfo('Waiting for MATLAB connection...')
    client_socket, addr = server_socket.accept()
    rospy.loginfo(f'Connection from {addr}')

    while not rospy.is_shutdown():
        data = client_socket.recv(1024)
        if not data:
            break

        # Process received data from MATLAB (e.g., control commands)
        command = data.decode('utf-8')
        rospy.loginfo(f'Received command: {command}')

        # Publish the received command to the ROS topic
        control_pub.publish(command)

        # Simulate sending sensor data back to MATLAB
        sensor_data = 42.0  # Placeholder sensor data
        client_socket.sendall(str(sensor_data).encode('utf-8'))

    client_socket.close()
    server_socket.close()

if __name__ == '__main__':
    try:
        tcp_server()
    except rospy.ROSInterruptException:
        pass
```

### 2. Run the Python TCP/IP Server Node:

To run the server, use the following command in your terminal:
```bash
rosrun your_package_name tcp_server.py
```

This server will now be waiting for MATLAB to connect and will handle communication.

---

## Step 4: Set Up MATLAB for TCP/IP Communication

Now that the ROS node is running a TCP/IP server, we can configure MATLAB to connect as a client.

### 1. Create a TCP/IP Client in MATLAB:

MATLAB will send control commands and receive sensor data from the Python-based ROS node.

```matlab
% Connect to the ROS TCP/IP server
t = tcpclient('192.168.1.10', 5000);  % Replace with your ROS IP address

% Send control commands to ROS via the TCP server
write(t, uint8('Move Forward'));

% Receive sensor data from ROS
data = read(t, t.BytesAvailable);
disp('Received sensor data from ROS:');
disp(char(data));
```

### 2. Run MATLAB Control Script:

Run the MATLAB script to establish a connection, send commands to the ROS system, and receive feedback.

---

## Step 5: Integrate MATLAB, Unity, Gazebo, and ROS

Now that the TCP/IP communication is established between MATLAB and ROS, the full system integration is as follows:

1. **Gazebo** simulates the BlueROV robot and environment.
2. **ROS** controls the BlueROV in Gazebo using the Python-based TCP/IP server.
3. **MATLAB** connects to ROS via TCP/IP to send control commands and receive sensor data.
4. **Unity** visualizes the BlueROV and can either communicate with ROS directly using ROS# or via custom TCP/IP communication.

---

## Conclusion

This guide demonstrates how to set up communication between **Gazebo**, **Unity**, **ROS**, **BlueROV**, and **MATLAB** using **ROS with Python and MATLAB via TCP/IP**. By using ROS as the central hub and utilizing TCP/IP communication, all components can exchange data effectively, allowing for complex simulations, control, and visualization setups.

Feel free to customize the TCP/IP communication to fit your specific project needs, and experiment with different control algorithms in MATLAB to interact with the BlueROV in Gazebo and Unity.

---

## References

- [ROS Documentation](http://wiki.ros.org/)
- [Gazebo Documentation](http://gazebosim.org/tutorials)
- [Unity ROS# Plugin](https://github.com/siemens/ros-sharp)

---
