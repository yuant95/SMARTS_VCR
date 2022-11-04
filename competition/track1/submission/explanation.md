# Explanation

1. Smoothed bezier curve used for path planning from the waypoints ahead of the ego car;
2. A trained neural network is used for classfying the action into collision, wrong way, safe etc. The data is collected offline. The input includes a bird view 256 x 256 image, ego position, ego heading, action, the future 5 waypoints. 
3. Each step, a action will be proposed from the bezier curve with speed of 1, then use the classfier to score the action based on how safe it is. 
4. If it is not safe, change the speed to 0.01. Otherwise, accept the action.
5. Repeat. 

GitHub link: https://github.com/yuant95/SMARTS_VCR