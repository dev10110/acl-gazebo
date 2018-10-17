import rosbag

with rosbag.Bag('office_buena.bag', 'w') as outbag:
    for topic, msg, t in rosbag.Bag('office.bag').read_messages():
        # This also replaces tf timestamps under the assumption 
        # that all transforms in the message share the same timestamp
        if topic == "/office_cloud":
            msg.header.frame_id='world'
            outbag.write("/cloud", msg)
            break
