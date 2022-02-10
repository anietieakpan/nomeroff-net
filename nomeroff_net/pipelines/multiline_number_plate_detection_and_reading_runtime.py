from nomeroff_net.pipelines.multiline_number_plate_detection_and_reading \
    import MultilineNumberPlateDetectionAndReading
from nomeroff_net.pipelines.base import RuntimePipeline


class MultilineNumberPlateDetectionAndReadingRuntime(MultilineNumberPlateDetectionAndReading,
                                                     RuntimePipeline):
    """
    Number Plate Detection and reading runtime
    """

    def __init__(self,
                 *args,
                 **kwargs):
        MultilineNumberPlateDetectionAndReadingRuntime.__init__(*args, **kwargs)
        RuntimePipeline.__init__(self, self.pipelines)