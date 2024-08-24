import matplotlib.pyplot as plt
import numpy as np

def bezier_curve(p0, p1, p2, p3, t):
    p0, p1, p2, p3 = np.array(p0), np.array(p1), np.array(p2), np.array(p3)
    x = (1 - t)**3 * p0[0] + 3 * (1 - t)**2 * t * p1[0] + 3 * (1 - t) * t**2 * p2[0] + t**3 * p3[0]
    y = (1 - t)**3 * p0[1] + 3 * (1 - t)**2 * t * p1[1] + 3 * (1 - t) * t**2 * p2[1] + t**3 * p3[1]
    return x, y

def draw_letter(ax, letter, offset_x=0, offset_y=0):
    t = np.linspace(0, 1, 100)

    if letter == 'A':
        p0, p1, p2, p3 = [-0.5, 0], [-0.25, 0.5], [-0.25, 0.5], [0, 1]
        x1, y1 = bezier_curve(p0, p1, p2, p3, t)

        p0, p1, p2, p3 = [0, 1], [0.25, 0.5], [0.25, 0.5], [0.5, 0]
        x2, y2 = bezier_curve(p0, p1, p2, p3, t)

        x_middle = np.linspace(-0.25, 0.25, 100)
        y_middle = np.ones_like(x_middle) * 0.5

        ax.plot(x1 + offset_x, y1 + offset_y, lw=2, color='blue')
        ax.plot(x2 + offset_x, y2 + offset_y, lw=2, color='blue')
        ax.plot(x_middle + offset_x, y_middle + offset_y, lw=2, color='blue')

    elif letter == 'B':
        x_vertical = np.ones_like(t) * -0.5
        y_vertical = np.linspace(0, 1, 100)

        p0, p1, p2, p3 = [-0.5, 1], [0.5, 1], [0.5, 0.75], [-0.5, 0.5]
        x_upper, y_upper = bezier_curve(p0, p1, p2, p3, t)

        p0, p1, p2, p3 = [-0.5, 0.5], [0.5, 0.5], [0.5, 0.25], [-0.5, 0]
        x_lower, y_lower = bezier_curve(p0, p1, p2, p3, t)

        ax.plot(x_vertical + offset_x, y_vertical + offset_y, lw=2, color='blue')
        ax.plot(x_upper + offset_x, y_upper + offset_y, lw=2, color='blue')
        ax.plot(x_lower + offset_x, y_lower + offset_y, lw=2, color='blue')

    elif letter == 'C':
        p0, p1, p2, p3 = [0.5, 1], [-0.5, 1], [-0.5, 0], [0.5, 0]
        x, y = bezier_curve(p0, p1, p2, p3, t)

        ax.plot(x + offset_x, y + offset_y, lw=2, color='blue')

    elif letter == 'D':
        x_vertical = np.ones_like(t) * -0.5
        y_vertical = np.linspace(0, 1, 100)

        p0, p1, p2, p3 = [-0.5, 1], [0.5, 1], [0.5, 0], [-0.5, 0]
        x_curve, y_curve = bezier_curve(p0, p1, p2, p3, t)

        ax.plot(x_vertical + offset_x, y_vertical + offset_y, lw=2, color='blue')
        ax.plot(x_curve + offset_x, y_curve + offset_y, lw=2, color='blue')

    elif letter == 'E':
        x_vertical = np.ones_like(t) * -0.5
        y_vertical = np.linspace(0, 1, 100)

        x_top = np.linspace(-0.5, 0.5, 100)
        y_top = np.ones_like(x_top) * 1

        x_middle = np.linspace(-0.5, 0.25, 100)
        y_middle = np.ones_like(x_middle) * 0.5

        x_bottom = np.linspace(-0.5, 0.5, 100)
        y_bottom = np.zeros_like(x_bottom)

        ax.plot(x_vertical + offset_x, y_vertical + offset_y, lw=2, color='blue')
        ax.plot(x_top + offset_x, y_top + offset_y, lw=2, color='blue')
        ax.plot(x_middle + offset_x, y_middle + offset_y, lw=2, color='blue')
        ax.plot(x_bottom + offset_x, y_bottom + offset_y, lw=2, color='blue')

    elif letter == 'F':
        x_vertical = np.ones_like(t) * -0.5
        y_vertical = np.linspace(0, 1, 100)

        x_top = np.linspace(-0.5, 0.5, 100)
        y_top = np.ones_like(x_top) * 1

        x_middle = np.linspace(-0.5, 0.25, 100)
        y_middle = np.ones_like(x_middle) * 0.5

        ax.plot(x_vertical + offset_x, y_vertical + offset_y, lw=2, color='blue')
        ax.plot(x_top + offset_x, y_top + offset_y, lw=2, color='blue')
        ax.plot(x_middle + offset_x, y_middle + offset_y, lw=2, color='blue')

    elif letter == 'G':
        p0, p1, p2, p3 = [0.5, 0], [-0.5, 0], [-0.5, 1], [0.5, 1]
        x1, y1 = bezier_curve(p0, p1, p2, p3, t)
        p0, p1, p2, p3 = [0, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]
        x2, y2 = bezier_curve(p0, p1, p2, p3, t)
        ax.plot(x1 + offset_x, y1 + offset_y, lw=2, color='blue')
        ax.plot(x2 + offset_x, y2 + offset_y, lw=2, color='blue')

    elif letter == 'H':
        x_vertical1 = np.ones_like(t) * -0.5
        x_vertical2 = np.ones_like(t) * 0.5
        y_vertical = np.linspace(0, 1, 100)

        x_middle = np.linspace(-0.5, 0.5, 100)
        y_middle = np.ones_like(x_middle) * 0.5

        ax.plot(x_vertical1 + offset_x, y_vertical + offset_y, lw=2, color='blue')
        ax.plot(x_vertical2 + offset_x, y_vertical + offset_y, lw=2, color='blue')
        ax.plot(x_middle + offset_x, y_middle + offset_y, lw=2, color='blue')

    elif letter == 'I':
        x_vertical = np.zeros_like(t)
        y_vertical = np.linspace(0, 1, 100)

        x_top_bottom = np.linspace(-0.25, 0.25, 100)
        y_top = np.ones_like(x_top_bottom)
        y_bottom = np.zeros_like(x_top_bottom)

        ax.plot(x_vertical + offset_x, y_vertical + offset_y, lw=2, color='blue')
        ax.plot(x_top_bottom + offset_x, y_top + offset_y, lw=2, color='blue')
        ax.plot(x_top_bottom + offset_x, y_bottom + offset_y, lw=2, color='blue')

    elif letter == 'J':
        p0, p1, p2, p3 = [0.5, 1], [0.5, 0.25], [0.25, 0], [0, 0]
        x_curve, y_curve = bezier_curve(p0, p1, p2, p3, t)

        x_vertical = np.zeros_like(t)
        y_vertical = np.linspace(0, 1, 100)

        ax.plot(x_curve + offset_x, y_curve + offset_y, lw=2, color='blue')
        ax.plot(x_vertical + offset_x, y_vertical + offset_y, lw=2, color='blue')

    elif letter == 'K':
        x_vertical = np.ones_like(t) * -0.5
        y_vertical = np.linspace(0, 1, 100)

        p0, p1, p2, p3 = [-0.5, 0.5], [0, 0.5], [0.25, 0.75], [0.5, 1]
        x_upper, y_upper = bezier_curve(p0, p1, p2, p3, t)

        p0, p1, p2, p3 = [-0.5, 0.5], [0, 0.5], [0.25, 0.25], [0.5, 0]
        x_lower, y_lower = bezier_curve(p0, p1, p2, p3, t)

        ax.plot(x_vertical + offset_x, y_vertical + offset_y, lw=2, color='blue')
        ax.plot(x_upper + offset_x, y_upper + offset_y, lw=2, color='blue')
        ax.plot(x_lower + offset_x, y_lower + offset_y, lw=2, color='blue')

    elif letter == 'L':
        x_vertical = np.ones_like(t) * -0.5
        y_vertical = np.linspace(0, 1, 100)

        x_bottom = np.linspace(-0.5, 0.5, 100)
        y_bottom = np.zeros_like(x_bottom)

        ax.plot(x_vertical + offset_x, y_vertical + offset_y, lw=2, color='blue')
        ax.plot(x_bottom + offset_x, y_bottom + offset_y, lw=2, color='blue')

    elif letter == 'M':
        p0, p1, p2, p3 = [-0.5, 0], [-0.25, 0.75], [-0.25, 0.75], [0, 1]
        x_left, y_left = bezier_curve(p0, p1, p2, p3, t)

        p0, p1, p2, p3 = [0, 1], [0.25, 0.75], [0.25, 0.75], [0.5, 0]
        x_right, y_right = bezier_curve(p0, p1, p2, p3, t)

        x_left_vertical = np.ones_like(t) * -0.5
        y_left_vertical = np.linspace(0, 1, 100)

        x_right_vertical = np.ones_like(t) * 0.5
        y_right_vertical = np.linspace(0, 1, 100)

        ax.plot(x_left + offset_x, y_left + offset_y, lw=2, color='blue')
        ax.plot(x_right + offset_x, y_right + offset_y, lw=2, color='blue')
        ax.plot(x_left_vertical + offset_x, y_left_vertical + offset_y, lw=2, color='blue')
        ax.plot(x_right_vertical + offset_x, y_right_vertical + offset_y, lw=2, color='blue')


    elif letter == 'N':
        h = 1  
        w = 0.5  
        
        p0_left = [0, 0]
        p1_left = [0, 0.33 * h]
        p2_left = [0, 0.66 * h]
        p3_left = [0, h]
        
        p0_right = [w, 0]
        p1_right = [w, 0.33 * h]
        p2_right = [w, 0.66 * h]
        p3_right = [w, h]
        
        p0_diag = [0, h]
        p1_diag = [0.33 * w, 0.66 * h]
        p2_diag = [0.66 * w, 0.33 * h]
        p3_diag = [w, 0]
        
        x_left, y_left = bezier_curve(p0_left, p1_left, p2_left, p3_left, t)
        ax.plot(x_left + offset_x, y_left + offset_y, lw=2, color='blue')

        x_right, y_right = bezier_curve(p0_right, p1_right, p2_right, p3_right, t)
        ax.plot(x_right + offset_x, y_right + offset_y, lw=2, color='blue')

        x_diag, y_diag = bezier_curve(p0_diag, p1_diag, p2_diag, p3_diag, t)
        ax.plot(x_diag + offset_x, y_diag + offset_y, lw=2, color='blue')
        
        ax.set_xlim(-0.1 + offset_x, 0.6 + offset_x)
        ax.set_ylim(-0.1 + offset_y, 1.1 + offset_y)
        ax.set_aspect('equal')
        ax.axis('on')

    elif letter == 'O':
        theta = np.linspace(0, 2 * np.pi, 100)
        x = 0.5*np.cos(theta)
        y = 0.5*np.sin(theta)
        ax.plot(x + offset_x, y + offset_y, lw=2, color='blue')

    elif letter == 'P':
        x_vertical = np.ones_like(t) * -0.5
        y_vertical = np.linspace(0, 1, 100)

        p0, p1, p2, p3 = [-0.5, 1], [0.5, 1], [0.5, 0.75], [-0.5, 0.5]
        x_upper, y_upper = bezier_curve(p0, p1, p2, p3, t)

        ax.plot(x_vertical + offset_x, y_vertical + offset_y, lw=2, color='blue')
        ax.plot(x_upper + offset_x, y_upper + offset_y, lw=2, color='blue')

    elif letter == 'Q':
        p0, p1, p2, p3 = [0.5, 1], [-0.5, 1], [-0.5, 0], [0.5, 0]
        x, y = bezier_curve(p0, p1, p2, p3, t)

        p0, p1, p2, p3 = [0.5, 0], [0.75, -0.25], [0.5, -0.5], [0, -0.5]
        x_diag, y_diag = bezier_curve(p0, p1, p2, p3, t)

        ax.plot(x + offset_x, y + offset_y, lw=2, color='blue')
        ax.plot(x_diag + offset_x, y_diag + offset_y, lw=2, color='blue')

    elif letter == 'R':
        x_vertical = np.ones_like(t) * -0.5
        y_vertical = np.linspace(0, 1, 100)

        p0, p1, p2, p3 = [-0.5, 1], [0.5, 1], [0.5, 0.75], [-0.5, 0.5]
        x_upper, y_upper = bezier_curve(p0, p1, p2, p3, t)

        p0, p1, p2, p3 = [-0.5, 0.5], [0, 0.5], [0.25, 0.25], [0.5, 0]
        x_diag, y_diag = bezier_curve(p0, p1, p2, p3, t)

        ax.plot(x_vertical + offset_x, y_vertical + offset_y, lw=2, color='blue')
        ax.plot(x_upper + offset_x, y_upper + offset_y, lw=2, color='blue')
        ax.plot(x_diag + offset_x, y_diag + offset_y, lw=2, color='blue')

    elif letter == 'S':
        r = 0.5  # Radius for the curves
        
        p0_upper = [r, 1]
        p1_upper = [-r, 1]
        p2_upper = [-r, 0.5]
        p3_upper = [r, 0.5]
        
        p0_lower = [-r, 0.5]
        p1_lower = [r, 0.5]
        p2_lower = [r, 0]
        p3_lower = [-r, 0]
        
        x_upper, y_upper = bezier_curve(p0_upper, p1_upper, p2_upper, p3_upper, t)
        ax.plot(x_upper + offset_x, y_upper + offset_y, lw=2, color='blue')
        
        x_lower, y_lower = bezier_curve(p0_lower, p1_lower, p2_lower, p3_lower, t)
        ax.plot(x_lower + offset_x, y_lower + offset_y, lw=2, color='blue')
        
        ax.set_xlim(-1 + offset_x, 1 + offset_x)
        ax.set_ylim(-0.5 + offset_y, 1.5 + offset_y)
        ax.set_aspect('equal')
        ax.axis('on')
    elif letter == 'T':
        x_vertical = np.zeros_like(t)
        y_vertical = np.linspace(0, 1, 100)

        x_top = np.linspace(-0.5, 0.5, 100)
        y_top = np.ones_like(x_top)

        ax.plot(x_vertical + offset_x, y_vertical + offset_y, lw=2, color='blue')
        ax.plot(x_top + offset_x, y_top + offset_y, lw=2, color='blue')

    elif letter == 'U':
        p0, p1, p2, p3 = [0.5, 1], [0.5, 0.5], [-0.5, 0.5], [-0.5, 1]
        x, y = bezier_curve(p0, p1, p2, p3, t)

        ax.plot(x + offset_x, y + offset_y, lw=2, color='blue')

    elif letter == 'V':
        p0, p1, p2, p3 = [-0.5, 1], [0, 0.25], [0, 0.25], [0.5, 1]
        x_left, y_left = bezier_curve(p0, p1, p2, p3, t)

        ax.plot(x_left + offset_x, y_left + offset_y, lw=2, color='blue')

    elif letter == 'W':
        p0, p1, p2, p3 = [-0.5, 1], [-0.25, 0.5], [-0.25, 0.5], [0, 0.75]
        x_left, y_left = bezier_curve(p0, p1, p2, p3, t)

        p0, p1, p2, p3 = [0, 0.75], [0.25, 0.5], [0.25, 0.5], [0.5, 1]
        x_right, y_right = bezier_curve(p0, p1, p2, p3, t)

        ax.plot(x_left + offset_x, y_left + offset_y, lw=2, color='blue')
        ax.plot(x_right + offset_x, y_right + offset_y, lw=2, color='blue')

    elif letter == 'X':
        p0, p1, p2, p3 = [-0.5, 0], [0, 0.5], [0, 0.5], [0.5, 1]
        x_diag1, y_diag1 = bezier_curve(p0, p1, p2, p3, t)

        p0, p1, p2, p3 = [-0.5, 1], [0, 0.5], [0, 0.5], [0.5, 0]
        x_diag2, y_diag2 = bezier_curve(p0, p1, p2, p3, t)

        ax.plot(x_diag1 + offset_x, y_diag1 + offset_y, lw=2, color='blue')
        ax.plot(x_diag2 + offset_x, y_diag2 + offset_y, lw=2, color='blue')

    elif letter == 'Y':
        p0, p1, p2, p3 = [-0.5, 1], [0, 0.75], [0, 0.75], [0.5, 1]
        x_upper, y_upper = bezier_curve(p0, p1, p2, p3, t)

        x_vertical = np.zeros_like(t)
        y_vertical = np.linspace(0, 0.5, 100)

        ax.plot(x_upper + offset_x, y_upper + offset_y, lw=2, color='blue')
        ax.plot(x_vertical + offset_x, y_vertical + offset_y, lw=2, color='blue')

    elif letter == 'Z':
        x_top = np.linspace(-0.5, 0.5, 100)
        y_top = np.ones_like(x_top)

        p0, p1, p2, p3 = [-0.5, 1], [0, 0.5], [0, 0.5], [0.5, 0]
        x_diag, y_diag = bezier_curve(p0, p1, p2, p3, t)

        x_bottom = np.linspace(-0.5, 0.5, 100)
        y_bottom = np.zeros_like(x_bottom)

        ax.plot(x_top + offset_x, y_top + offset_y, lw=2, color='blue')
        ax.plot(x_diag + offset_x, y_diag + offset_y, lw=2, color='blue')
        ax.plot(x_bottom + offset_x, y_bottom + offset_y, lw=2, color='blue')

def write_text(ax, text, start_x=0, start_y=0, letter_spacing=3):
    x_offset = start_x
    for letter in text:
        if letter.isalpha():
            draw_letter(ax, letter.upper(), offset_x=x_offset, offset_y=start_y)
            x_offset += letter_spacing  # Adjust spacing for better separation
        elif letter.isspace():
            x_offset += 5 # Space for spaces
        else:
            x_offset += letter_spacing / 2  # Adjust for non-alphabetic characters (e.g., spaces)

plt.figure(figsize=(12, 6))
ax = plt.gca()
write_text(ax, 'GIFT FROM SHIVSHARAN', start_x=0, start_y=0, letter_spacing=4)  # Increased spacing
plt.axis('equal')
plt.grid(True)
plt.show()
