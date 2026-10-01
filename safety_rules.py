def analyze_safety(detections):
    """
    Analyze PPE detections and identify safety violations.

    Args:
        detections: List of detected class names.

    Returns:
        List of detected safety incidents.
    """

    incidents = []

    # Missing helmet
    if "Person" in detections and "no_helmet" in detections:
        incidents.append({
            "type": "Missing Helmet",
            "severity": "HIGH",
            "message": "Worker detected without a safety helmet."
        })

    # Missing goggles
    if "Person" in detections and "no_goggle" in detections:
        incidents.append({
            "type": "Missing Goggles",
            "severity": "MEDIUM",
            "message": "Worker detected without safety goggles."
        })

    # Missing gloves
    if "Person" in detections and "no_gloves" in detections:
        incidents.append({
            "type": "Missing Gloves",
            "severity": "MEDIUM",
            "message": "Worker detected without safety gloves."
        })

    # Missing boots
    if "Person" in detections and "no_boots" in detections:
        incidents.append({
            "type": "Missing Boots",
            "severity": "MEDIUM",
            "message": "Worker detected without safety boots."
        })

    # No PPE
    if "Person" in detections and "none" in detections:
        incidents.append({
            "type": "No PPE",
            "severity": "CRITICAL",
            "message": "Worker detected without required PPE."
        })

    return incidents