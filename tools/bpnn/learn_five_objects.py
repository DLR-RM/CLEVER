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
import copy
import time

from clever.clever import TheMostCleverSystem
from clever.utils import WebcamImageProvider

from prompt_toolkit import prompt, PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit import print_formatted_text

__author__ = "lee_jn"
__copyright__ = "lee_jn"
__license__ = "MIT"

style = Style.from_dict({
    '':         '#00aa00',
    'username': '#884444',
    'at':       '#00aa00',
    'colon':    '#0000aa',
    'pound':    '#00aa00',
    'host':     '#00ffff bg:#444400',
    'path':     'ansicyan underline',
})

user_commands = ["Can you understand what's on the table? \n",
                 "Yes, very good CLEVER!. \n",
                 "No, but I give you instructions. \n",
                 "Yes, I can help you. \n",
                 "Yes, go ahead. \n",
                 "I cannot recognize any objects with enough confidence. Can you help me? \n",]


def main(args):
    # set the device if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initiate the clever system
    clever = TheMostCleverSystem(args=args, device=device)
    
    # initiate a promt session
    session = PromptSession()

    # initiate an image provider
    image_provider = WebcamImageProvider()

    # perform semantic segmentation and print out the objects.
    clever.assign_bbox([[80, 40], [560, 400]])
    clever.assign_references(image_acqusition=image_provider)
    buffer = clever.buffering(image_acqusition=image_provider, buffer_size=200)
    cv2.destroyAllWindows()

    # looping
    for i in range(0, 1):
        true_object_name = session.prompt('CLEVER: What is this object? \n Human: ', style=style)
        print_formatted_text('CLEVER: Please show me a/an {}. \n'.format(true_object_name), style=style)
        
        clever.assign_bbox([[80, 40], [560, 400]])
        clever.assign_references(image_acqusition=image_provider)

        clever.data_collection(image_acqusition=image_provider, object_name=true_object_name, speech_engine=None)
        print_formatted_text('CLEVER: Thank you. I start training! \n', style=style)        
        start_time = time.time()            
        clever.active_learn(object_name=true_object_name, random_selection=False)
        print_formatted_text('CLEVER: I finished learning the {}! \n'.format(true_object_name), style=style)
        print_formatted_text('CLEVER: It took me {} seconds! \n'.format(time.time() - start_time), style=style)
        cv2.destroyAllWindows()

    # perform semantic segmentation and print out the objects.
    clever.assign_bbox([[80, 40], [560, 400]])
    clever.assign_references(image_acqusition=image_provider)
    buffer = clever.buffering(image_acqusition=image_provider, buffer_size=250)

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser("Demonstration of TheMostCleverSystem")
    parser.add_argument("--model_path", help="Path to the trained model", required=True)
    parser.add_argument("--pooldata_dir", help="Path to the dataset location", required=True)
    parser.add_argument("--model_pt_file", type=str, default=None, help="Path to the model pt file")
    parser.add_argument("--prior_pt_file", type=str, default=None, help="Path to the prior pt file")

    args = parser.parse_args()

    main(args)
