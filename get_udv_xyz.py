import rospy
from std_msgs.msg import Float32MultiArray
from tf import TransformListener
import tf
import signal

#Global variables
b_running = True
uvd = [-1, -1, -1]
b_read = False

def cb(data):
    global uvd, b_read
    #print(rospy.get_caller_id(), " : ")
    uvd = data.data
    b_read = True
    #rospy.loginfo(uvd)  


def handler(signum, frame):
    rospy.loginfo("STOP signal was received! (Ctr+C)")
    global b_running
    b_running = False


#------------MAIN----------------------
rospy.init_node('listener_test')
listener = TransformListener()
rate = rospy.Rate(1) #HZ
signal.signal(signal.SIGINT, handler)
rospy.Subscriber("uvd", Float32MultiArray, cb)
trans = [-1, -1, -1]
source_link = '/world'
dest_link = '/right_hand'
while b_running:
    try:
        #t = listener.getLatestCommonTime(source_link, dest_link)
        #print('t = ', t)
        (trans,rot) = listener.lookupTransform(source_link, dest_link, rospy.Time(0))
    except tf.Exception as e :
        print(e)
        trans = [0, 0, 0]
        rot = [0, 0, 0, 0]
    rate.sleep()
    # euler = tf.transformations.euler_from_quaternion(rot)
    if b_read:
        b_read = False
        #print('trans = ', trans)
        print(f'({uvd[0]:.2f}, {uvd[1]:.2f}, {uvd[2]:.2f}) -> ({trans[0]:.2f}, {trans[1]:.2f}, {trans[2]:.2f})')
# rospy.spin()