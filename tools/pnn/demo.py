""" Test scripts to integrate for demonstration scenario.
Scenario 1: An apple or banana on the table.

ARMAR: At your service.
HUMAN: Can you understand what's on the table?
ARMAR: Let me show you (opening the visualization).
ARMAR: Yes, I see a XX with around X% confidence. (after collecting the amount of predictions)
HUMAN: Very good.
ARMAR: At your service.

Scenario 2: Closing sim2real gap.

ARMAR: At your service.
HUMAN: Can you understand what's on the table?
ARMAR: Let me show you (opening the visualization).
ARMAR: No, I am unsure about this object. Can you help me?
HUMAN: Yes, I can help you (segmentation gui opens and human saves images).
HUMAN: This is a banana. 
ARMAR: Let me learn it. Counting...
ARMAR: Learning completed.
ARMAR: At your service.
HUMAN: Can you understand what's on the table?
ARMAR: Let me show you (opening the visualization).
ARMAR: Yes, I see a XX with around X% confidence. Let me show you.
HUMAN: Very good.
ARMAR: At your service.

Scenario 3: Deformed banana on the table.

ARMAR: At your service.
HUMAN: Can you understand what's on the table?
ARMAR: Let me show you (opening the visualization).
ARMAR: No, I am unsure about this object. Can you help me?
HUMAN: Yes, I can help you (segmentation gui opens and human saves images).
HUMAN: This is a banana. 
ARMAR: Let me learn it. Counting..
ARMAR: Learning completed.
ARMAR: At your service.
HUMAN: Can you understand what's on the table?
ARMAR: Let me show you (opening the visualization).
ARMAR: Yes, I see a XX with around X% confidence. Let me show you.
HUMAN: Very good.
ARMAR: At your service.

We can also repeat this for the objects.
In these scenarios, we only have banana and apple.
Data should be collected however for the benchmark sets.
"""

import argparse
import torch
import numpy as np
import cv2

from clever.clever import TheCleverSystem
from clever.utils import WebcamImageProvider

from prompt_toolkit import prompt, PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit import print_formatted_text

__author__ = "lee_jn"
__copyright__ = "lee_jn"
__license__ = "MIT"

style = Style.from_dict({
    '':          '#00aa00',
    'username': '#884444',
    'at':       '#00aa00',
    'colon':    '#0000aa',
    'pound':    '#00aa00',
    'host':     '#00ffff bg:#444400',
    'path':     'ansicyan underline',
})

user_commands = ["Can you understand what's on the table? \n",
                 "Yes, very good ARMAR!. \n",
                 "No, but I give you instructions. \n",
                 "Yes, I can help you. \n",
                 "Yes, go ahead. \n"]

def display_images_with_center(image_provider):
    while True:
        image = image_provider.return_image()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        input_point = np.array([[int(image.shape[0]/2), int(image.shape[1]/2)]])
        image = cv2.circle(
            image, (int(image.shape[1]/2), int(image.shape[0]/2)), 
            radius=10, color=(0, 0, 0), thickness=-1)
        cv2.imshow('Object alignment', image)

        pressedKey = cv2.waitKey(1) & 0xFF
        if pressedKey == ord('q'):
            break
    cv2.destroyAllWindows()


def main(args):
    # set the device if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initiate the clever system
    clever = TheCleverSystem(args=args, device=device)

    # initiate a promt session
    session = PromptSession()

    # initiate an image provider (NOTE: we can select armarx image provider here)
    image_provider = WebcamImageProvider()

    # demonstration loop
    while True:
        text1 = session.prompt('ARMAR: At your service. \n  Human: ', style=style, 
                               completer=WordCompleter([user_commands[0]]))
        if text1 == user_commands[0]:
            # perform semantic segmentation and print out the objects.
            buffer = clever.buffering(image_acqusition=image_provider, buffer_size=200)

            # perform query step and make decisions.            
            is_query, discovered_objects = clever.query_decisions(buffer)

            print(discovered_objects)
            print(is_query)
            if not any(is_query):
                count = 0
                for confidence in discovered_objects['confidence']:
                    if confidence > 0.75:
                        print_formatted_text('ARMAR: Yes, I see {} with around {}% confidence. \n'.format(discovered_objects['objects'][count], confidence), style=style)
                    count = count + 1
                
                while True:
                    text2 = session.prompt('ARMAR: Did I get it correct?. \n  Human: ', 
                            style=style, 
                            completer=WordCompleter([user_commands[1], user_commands[2]]))
                    if text2 == user_commands[1]:
                        break
                    elif text2 == user_commands[2]:
                        # training with the objects
                        true_object_name = session.prompt('ARMAR: This is a/an?. \n  Human: ', 
                                                          style=style)
                        print_formatted_text('ARMAR: Can you show me a/an {}. \n'.format(true_object_name), style=style)
                        display_images_with_center(image_provider)
                        clever.data_collection(image_acqusition=image_provider, object_name=true_object_name)

                        session.prompt('ARMAR: Should I start training?. \n  Human: ', 
                                        style=style, 
                                        completer=WordCompleter([user_commands[4]]))
                        
                        print_formatted_text('ARMAR: Ok, I start training! \n', style=style)
                        clever.active_learn(object_name=true_object_name)
                        print_formatted_text('ARMAR: I finished learning a/an {}! \n'.format(true_object_name), style=style)

                        # training the wrong predictions
                        trained_objects = []
                        for object, confidence in zip(discovered_objects['objects'], discovered_objects['confidence']):
                            if confidence > 0.75 and object != true_object_name and object not in trained_objects:
                                print_formatted_text('ARMAR: Can you show me a/an {}. \n'.format(object), style=style)
                                display_images_with_center(image_provider)
                                clever.data_collection(image_acqusition=image_provider, object_name=object)

                                session.prompt('ARMAR: Should I start training?. \n  Human: ', 
                                               style=style, 
                                               completer=WordCompleter([user_commands[4]]))
                                
                                print_formatted_text('ARMAR: Ok, I start training! \n', style=style)
                                clever.active_learn(object_name=object)
                                print_formatted_text('ARMAR: I finished learning a/an {}! \n'.format(object), style=style)
                                trained_objects.append(object)
                        break
                    else:
                        print_formatted_text('ARMAR: I did not understand you. \n', style=style)
            else:
                text3 = session.prompt('ARMAR: No, I am unsure about this object. Can you help me?. \n Human: ',
                                       style=style,
                                       completer=WordCompleter([user_commands[3]]))
                if text3 == user_commands[3]:
                        # training with the objects
                        true_object_name = session.prompt('ARMAR: This is a/an?. \n  Human: ', 
                                                          style=style)
                        print_formatted_text('ARMAR: Can you show me a/an {}. \n'.format(true_object_name), style=style)
                        display_images_with_center(image_provider)
                        clever.data_collection(image_acqusition=image_provider, object_name=true_object_name)

                        session.prompt('ARMAR: Should I start training?. \n  Human: ', 
                                        style=style, 
                                        completer=WordCompleter([user_commands[4]]))
                        
                        print_formatted_text('ARMAR: Ok, I start training! \n', style=style)
                        clever.active_learn(object_name=true_object_name)
                        print_formatted_text('ARMAR: I finished learning a/an {}! \n'.format(true_object_name), style=style)

                        # training the wrong predictions
                        trained_objects = []
                        for object, confidence in zip(discovered_objects['objects'], discovered_objects['confidence']):
                            if confidence > 0.75 and object != true_object_name and object not in trained_objects:
                                print_formatted_text('ARMAR: Can you show me a/an {}. \n'.format(object), style=style)
                                display_images_with_center(image_provider)
                                clever.data_collection(image_acqusition=image_provider, object_name=object)

                                session.prompt('ARMAR: Should I start training?. \n  Human: ', 
                                               style=style, 
                                               completer=WordCompleter([user_commands[4]]))
                                
                                print_formatted_text('ARMAR: Ok, I start training! \n', style=style)
                                clever.active_learn(object_name=object)
                                print_formatted_text('ARMAR: I finished learning a/an {}! \n'.format(object), style=style)
                                trained_objects.append(object)
                else:
                    print_formatted_text('ARMAR: Ok, then next time! \n', style=style)
        else:
            print_formatted_text('ARMAR: I did not understand you. \n', style=style)


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser("Loads images and labels of HOWS for a given sequence")
    parser.add_argument("--model_path", help="Path to the trained model", required=True)
    parser.add_argument("--pooldata_dir", help="Path to the dataset location", required=True)
    args = parser.parse_args()

    main(args)