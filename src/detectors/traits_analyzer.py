class TraitsAnalyzer:
    """
    Simple personality trait analyzer based on emotion distribution.
    You can later expand this to use a trained ML model.
    """

    def infer_traits(self, emotions_dict):
        if not emotions_dict:
            return "Unknown"

        happiness = emotions_dict.get('happy', 0)
        anger = emotions_dict.get('angry', 0)
        sadness = emotions_dict.get('sad', 0)
        surprise = emotions_dict.get('surprise', 0)

        if happiness > 0.5:
            return "Optimistic / Cheerful"
        elif anger > 0.5:
            return "Stressed / Irritated"
        elif sadness > 0.5:
            return "Low energy / Reflective"
        elif surprise > 0.5:
            return "Alert / Curious"
        else:
            return "Neutral / Calm"
