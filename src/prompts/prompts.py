SAMPLING_WINDOW_TEMPLATE_v2 = """You are an expert video analyst. You are given a video with a total duration of [DURATION] seconds.

Your task is to identify at least [MAX_WINDOW] consecutive, non-overlapping temporal segments and provide a clear natural-language description of what happens in each segment.

Output your thought process briefly within the <think> </think> tags.

List the start and end times in the format [[s1, e1], [s2, e2], ...] (in seconds, precise to two decimal places) within the <time> </time> tags.
The segments should be non-overlapping and ordered chronologically, and have a reasonable duration (at least 3 seconds, but not the entire video)

Then, for each segment, write a event-level description explaining what occurs during that time. Each description should be a complete sentence and mention the main action(s) and participants. Please avoid generic or image-style captions

The number of segments in <time></time> and the number of descriptions in <description></description> must be exactly the same.

Output format (STRICT):
<think>...</think>
<time>[[s1, e1], [s2, e2], ..., [s_n, e_n]]</time>
<description>["description for segment 1", "description for segment 2", ..., "description for segment n"]</description>
"""

QUESTION_TEMPLATE_TG_v1 = """To accurately pinpoint the event "[EVENT]" in the video, determine the precise time period of the event.

Output your thought process within the <think> </think> tags, including analysis with either specific time ranges (xx.xx to xx.xx) in <answer> </answer> tags.

Then, provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example: "12.54 to 17.83"."""
