#!/usr/bin/env python3
from preparation import Preparation
from Main_stream import Frame_stream
from conf import get_args

import time

def main():
    args = get_args()
    if args.video:
        preparation = Preparation(video_file=args.video)
    else:
        preparation = Preparation(video_source=0)

    while not (preparation.is_started() or preparation.is_done()):
        time.sleep(1)

    if preparation.is_done():
        return

    frame_stream = Frame_stream()
    frame_stream.frames_flow(preparation.get_flow())

    try:

        frame_stream.start()

        while True:
            time.sleep(0.01)
            if frame_stream.is_done() or preparation.is_done():
                raise SystemExit(0)

    except (KeyboardInterrupt, SystemExit):
        preparation.stop()
        frame_stream.stop()
        preparation.join()


if __name__ == '__main__':
    main()
