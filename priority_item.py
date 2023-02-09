class PriorityItem(object):
    """
    storing information of a node in the search tree
    the sequence of filters and transformations used to arrive at an abstracted graph is stored as the data,
    instead of the abstracted graphs themselves
    """
    def __init__(self, data, abstraction, priority, secondary_priority=0):
        self.data = data
        self.abstraction = abstraction
        self.priority = priority
        self.secondary_priority = secondary_priority

    def __lt__(self, other):
        if self.priority == other.priority:
            if self.secondary_priority == other.secondary_priority:
                return len(self.data[0]["filters"]) < len(other.data[0]["filters"])
            else:
                return self.secondary_priority < other.secondary_priority
        else:
            return self.priority < other.priority
