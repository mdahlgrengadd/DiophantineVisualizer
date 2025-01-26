import streamlit as st
import numpy as np
import plotly.graph_objects as go
from math import gcd, sqrt
from typing import Tuple, List

def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Extended Euclidean Algorithm
    Returns (d, x, y) where d = gcd(a, b) and ax + by = d
    """
    if b == 0:
        return a, 1, 0
    d, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return d, x, y

def find_particular_solution(a: int, b: int, c: int) -> Tuple[int, int]:
    """
    Find a particular solution to the Diophantine equation ax + by = c
    Returns (x0, y0) where ax0 + by0 = c
    """
    d, x0, y0 = extended_gcd(abs(a), abs(b))
    if c % d != 0:
        raise ValueError(f"No integer solutions exist: gcd({a}, {b}) = {d} does not divide {c}")
    
    x0 *= c // d
    y0 *= c // d
    
    if a < 0:
        x0 = -x0
    if b < 0:
        y0 = -y0
    
    return x0, y0

def find_min_t_range(a: int, b: int, c: int, target_x=None, target_y=None) -> int:
    """
    Calculate minimum t range needed to include solutions near origin
    and optionally a target solution (x,y)
    """
    x0, y0 = find_particular_solution(a, b, c)
    d = gcd(a, b)
    
    # Calculate t for origin-nearest solution
    if target_x is not None and target_y is not None:
        # Calculate t needed for target solution
        t = (target_x - x0) // (b // d)
        return max(abs(t) + 5, 50)  # Add buffer and minimum range
    
    # Calculate rough estimate of t needed for reasonable coverage
    t_estimate = max(abs(x0 // (b // d)), abs(y0 // (a // d))) + 5
    return max(t_estimate, 50)  # Minimum range of 50

def generate_solutions(a: int, b: int, c: int, t_range: int, max_solutions: int = 20) -> List[Tuple[int, int, int]]:
    """
    Generate solutions sorted by distance from origin, limited to max_solutions
    """
    try:
        x0, y0 = find_particular_solution(a, b, c)
    except ValueError:
        return []
    
    d = gcd(a, b)
    solutions = []
    
    for t in range(-t_range, t_range + 1):
        x = x0 + (b // d) * t
        y = y0 - (a // d) * t
        dist = sqrt(x*x + y*y)  # Distance using actual coordinates
        solutions.append((x, y, t, dist))
    
    # Sort solutions by distance from origin
    solutions.sort(key=lambda x: x[3])
    # Take only max_solutions closest to origin
    solutions = solutions[:max_solutions]
    return [(x, y, t) for x, y, t, _ in solutions]

def calculate_window_size(solutions: List[Tuple[int, int, int, float, int, int]]) -> int:
    """
    Calculate appropriate window size based on the solutions
    """
    if not solutions:
        return 10
    
    # Get maximum absolute coordinates from plot values
    max_coord = max(max(abs(s[4]), abs(s[5])) for s in solutions)
    # Add 20% padding
    return int(max_coord * 1.2)

def create_plot(a: int, b: int, c: int, solutions: List[Tuple[int, int, int]]):
    """
    Create an interactive plot showing the actual solution points
    """
    if not solutions:
        return go.Figure()
    
    # Get x, y coordinates for plotting
    x_coords = [s[0] for s in solutions]
    y_coords = [s[1] for s in solutions]
    
    # Calculate window size based on actual coordinates
    max_coord = max(max(abs(x) for x in x_coords), max(abs(y) for y in y_coords))
    window_size = int(max_coord * 1.2)  # Add 20% padding
    
    # Create figure
    fig = go.Figure()
    
    # Add prominent axes
    fig.add_trace(go.Scatter(
        x=[-window_size, window_size],
        y=[0, 0],
        mode='lines',
        name='X-axis',
        line=dict(color='black', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 0],
        y=[-window_size, window_size],
        mode='lines',
        name='Y-axis',
        line=dict(color='black', width=2)
    ))
    
    # Add the equation line
    x = np.linspace(-window_size, window_size, 1000)
    y = (c - a * x) / b if b != 0 else np.full_like(x, c/a if a != 0 else np.nan)
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        name=f'{a}x + {b}y = {c}',
        line=dict(color='blue', width=2)
    ))

    # Add integer solutions with hover text
    hover_text = [f'Solution for t={t}:<br>x={x}, y={y}' 
                 for x, y, t in solutions]
    
    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords,
        mode='markers',
        name='Integer Solutions',
        text=hover_text,
        hoverinfo='text',
        marker=dict(
            size=10,
            color='red',
            symbol='circle'
        )
    ))
    
    # Update layout with quadrant labels
    fig.update_layout(
        title=f'Diophantine Equation: {a}x + {b}y = {c}',
        xaxis_title='x',
        yaxis_title='y',
        xaxis=dict(
            range=[-window_size, window_size],
            showgrid=True,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black'
        ),
        yaxis=dict(
            range=[-window_size, window_size],
            showgrid=True,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black'
        ),
        showlegend=True,
        annotations=[
            dict(x=window_size*0.9, y=window_size*0.9, text="Quadrant I (+,+)", showarrow=False),
            dict(x=-window_size*0.9, y=window_size*0.9, text="Quadrant II (-,+)", showarrow=False),
            dict(x=-window_size*0.9, y=-window_size*0.9, text="Quadrant III (-,-)", showarrow=False),
            dict(x=window_size*0.9, y=-window_size*0.9, text="Quadrant IV (+,-)", showarrow=False)
        ]
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    return fig

def main():
    try:
        # Set page to wide mode
        st.set_page_config(layout="wide")
        
        st.title("Diophantine Equation Visualizer")
        st.write("Explore integer solutions to equations of the form ax + by = c")
        
        # Create two columns for the layout
        left_col, right_col = st.columns([1, 3])
        
        # Input parameters in the left column
        with left_col:
            st.subheader("Parameters")
            a = st.slider("Select a", -100, 100, 2, 1)
            b = st.slider("Select b", -100, 100, 3, 1)
            c = st.number_input("Enter c", value=6, step=1)
            max_solutions = st.slider("Maximum solutions to display", 5, 50, 20)
            t_range = find_min_t_range(a, b, c)
        
        # Generate and display solutions in the right column
        with right_col:
            d = gcd(a, b)
            if c % d != 0:
                st.error(f"No integer solutions exist: gcd({a}, {b}) = {d} does not divide {c}")
                return

            solutions = generate_solutions(a, b, c, t_range, max_solutions)
            
            # Create and display plot
            fig = create_plot(a, b, c, solutions)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display solutions
            st.write("### Solutions (sorted by distance from origin)")
            solution_table = []
            for x, y, t in solutions:
                quadrant = "I (+,+)" if x >= 0 and y >= 0 else \
                          "II (-,+)" if x < 0 and y >= 0 else \
                          "III (-,-)" if x < 0 and y < 0 else \
                          "IV (+,-)"
                solution_table.append({
                    "t": t,
                    "x": x,
                    "y": y,
                    "Distance from (0,0)": round(sqrt(x*x + y*y), 2),
                    "Quadrant": quadrant
                })
            st.table(solution_table)
            
            # Mathematical explanation
            st.write("### Mathematical Details")
            x0, y0 = solutions[0][:2]  # Get the closest solution to origin
            st.write(f"- GCD(a, b) = {d}")
            st.write(f"- Particular solution closest to origin: x₀ = {x0}, y₀ = {y0}")
            st.write(f"- General solution form:")
            st.write(f"  x = {x0} + {b//d}t")
            st.write(f"  y = {y0} - {a//d}t")
            st.write("  where t is any integer")
                    
            # Add explanation about plot coordinates
            st.write("### Plot Coordinates Explanation")
            st.write(f"The plot shows the equation {a}x + {b}y = {c} in the coordinate plane. "
                    f"Each point (x, y) from the solution table corresponds to a plot point (ax, by) on the graph. "
                    f"This helps visualize how the solutions satisfy the equation, as each plot point lies on the line.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()