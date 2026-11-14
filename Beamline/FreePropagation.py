from Beamline.PropagationMatrix import PropagationMatrix

class FreePropagation(PropagationMatrix):
    """
    Implementation of a free space propagation matrix
    """
    def __init__(self, length: float) -> None:
        super().__init__()
        self.ABCD[0,0] = 1.0
        self.ABCD[0,1] = length
        self.ABCD[1,0] = 0.0
        self.ABCD[1,1] = 1.0
