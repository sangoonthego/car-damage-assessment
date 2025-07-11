class SeverityLevel:
    def __init__(self, class_name):
        self.class_name = class_name.lower().strip()
        self.severity = self.extract_from_class()

    def extract_from_class(self):
        parts = self.class_name.split("_")
        for level in ["minor", "moderate", "severe"]:
            if level in parts:
                return level
        return "unknown"
    
    def to_dict(self):
        return {
            "class": self.class_names,
            "severity": self.severity
        }
    
    def __str__(self):
        return f"{self.class_name} (Severity: {self.severity})"