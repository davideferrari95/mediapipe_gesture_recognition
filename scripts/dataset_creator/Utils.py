import rospy

# Countdown Function
def Start_countdown(num_of_secs):

    print("\nAcquisition Starts in:")

    # Wait Until 0 Seconds Remaining
    while (not rospy.is_shutdown() and num_of_secs != 0):
        m, s = divmod(num_of_secs, 60)
        min_sec_format = '{:02d}:{:02d}'.format(m, s)
        print(min_sec_format)
        rospy.sleep(1)
        num_of_secs -= 1

    print("\nSTART\n")

def rest_countdown(num_of_secs):


    # Wait Until 0 Seconds Remaining
    while (not rospy.is_shutdown() and num_of_secs != 0):
        m, s = divmod(num_of_secs, 60)
        min_sec_format = '{:02d}:{:02d}'.format(m, s)
        print(min_sec_format)
        rospy.sleep(1)
        num_of_secs -= 1

