import random

def generate_rect_files(n):
    def generate_random_rect():
        # Generate random x, y, width, and height within the specified ranges
        x = random.randint(-30, 50)
        y = random.randint(-50, 30)
        width = random.randint(3, 45)
        height = random.randint(3, 45)
        return [x, y, width, height]

    def check_overlap(rect1, rect2):
        # Unpack rectangles
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        # Check for overlap using axis-aligned bounding box (AABB) collision detection
        if (x1 < x2 + w2 and x1 + w1 > x2 and
            y1 < y2 + h2 and y1 + h1 > y2):
            return True
        
        return False

    def generate_rectangles():
        # Randomly choose how many rectangles to generate (between 4 and 15)
        num_rects = random.randint(3, 7)
        rectangles = []

        # Try to generate non-overlapping rectangles
        while len(rectangles) < num_rects:
            new_rect = generate_random_rect()
            if all(not check_overlap(new_rect, existing_rect) for existing_rect in rectangles):
                if not check_overlap(new_rect, [-10, -10, 20, 20]):
                    rectangles.append(new_rect)

        return rectangles

    def format_as_matrix(rectangles):
        # Convert the list of rectangles into a space-separated matrix
        return "\n".join(" ".join(map(str, rect)) for rect in rectangles)

    for i in range(1, n + 1):
        rectangles = generate_rectangles()
        matrix = format_as_matrix(rectangles)
        # Write to file (file name as 'rectangles_X.txt' where X is the file number)
        file_name = f"rects/rectangles_{i}"
        with open(file_name, "w") as file:
            file.write(matrix)
        print(f"Generated {file_name}")

# Example usage
generate_rect_files(500)  # Generates 500 files with random non-overlapping rectangles
