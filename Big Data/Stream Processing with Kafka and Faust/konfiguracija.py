
import confluent_kafka as kafka, socket

import os, socket




producer = kafka.Producer({'bootstrap.servers': "localhost:29092",
                  'client.id': socket.gethostname()})
                  
 consumer = kafka.Consumer({'bootstrap.servers': "localhost:29092",
                            'client.id': socket.gethostname(),
                             'group.id': 'test_group', 
                             'auto.offset.reset': 'earliest})

