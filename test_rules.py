from safety_rules import analyze_safety


# Test 1: Safe worker
detections = ["Person", "helmet", "vest", "gloves", "boots"]

incidents = analyze_safety(detections)

print("TEST 1 - Safe Worker")
print("Incidents:", incidents)


# Test 2: Missing helmet
detections = ["Person", "no_helmet", "vest"]

incidents = analyze_safety(detections)

print("\nTEST 2 - Missing Helmet")
print("Incidents:", incidents)


# Test 3: No PPE
detections = ["Person", "none"]

incidents = analyze_safety(detections)

print("\nTEST 3 - No PPE")
print("Incidents:", incidents)