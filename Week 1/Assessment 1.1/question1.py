#!/usr/bin/env python
### Assessment 1.1 Q1 v1 | Orrin Hatch ###

from typing import Dict, List


# --------- Question 1.1 ---------- #


def list_to_dict(input_list: list) -> dict:
    """
    Converts a list to a dictionary with word lengths as values

    Takes a list of strings and converts it to a dictionary where the keys
    are the strings and the values are the length of the strings.

    Parameters
    ----------
    input_list
        a list of strings

    Returns
    -------
    out_dict
        Dict[str, int]
    """

    # declare the output type
    out_dict: Dict[str, int] = {}

    for word in input_list:
        out_dict[word] = len(word)

    return out_dict


# --------- Question 1.2 ---------- #


def count_population(pop_dict: Dict[str, int], countries: List[str]) -> int:
    """
    Counts the total population of a list of countries

    Takes a `Dict[str, int]` (a dictionary with strings as `keys` and integers as
    `values`) and a `List[str]` (a list of strings) and returns an `int` where:

    - The dictionary contains countries as the `key` and the corresponding population
        as the `value`
    - The list contains the countries whose populations need to be summed

    Parameters
    ----------
    pop_dict
        a dictionary with countries as keys and population as values
    countries
        a list of countries whose population needs to be summed

    Returns
    -------
    total_population
        the sum of the population

    """

    ## For loops are costly and a tiny bit slow... has to go through every iteration.
    ## This is the "correct" version, but I am gonna experiment a little bit.

    # declare the output type
    # total_population: int = 0

    # for country in countries:
    #     total_population += pop_dict.get(country, 0)

    # return total_population

    # Generator expression inside sum() avoids building an intermediate list in
    # memory, making this speedier and more memory efficient than a 'for loop'. The
    # "extra" guard 'if c in pop_dict' will skip countries not in the dictionary
    # already rather than raising a KeyError or checking every country in the list.

    return sum(pop_dict[c] for c in countries if c in pop_dict)


# --------- Question 1.3 ---------- #


class Robot:
    """
    A class that holds basic details about a manipulator robot.

    The `init` method of the class stores each of the input arguments
    as a class attribute.

    Parameters
    ----------
    name
        the name of the robot
    n
        the number of joints in the robot
    joint_type
        the joint type of the robot, either `mixed`, `revolute`,
        or `prismatic`
    price
        the price of the robot

    """

    def __init__(self, name: str, n: int, joint_type: str, price: float):
        self.name = name
        self.n = n
        self.joint_type = joint_type
        self.price = price

    def __str__(self):
        """
        This method converts this instance of the `Robot` class into a string

        This method should return a string in the form:

        "{name} Robot Description: number of joints: {n}, joint type: {joint_type}

        Returns
        -------
        str
            A string representation of this `Robot`

        """
        return f"{self.name} Robot Description: number of joints: {self.n}, joint type: {self.joint_type}"


# --------- Question 1.4 ---------- #


def csv_to_robots(file_name: str) -> List[Robot]:
    """
    Converts a csv file to a list of `Robot` instances

    Takes a csv file and converts it to a list of `Robot` instances.
    The returned list should be a list of `Robot` classes using the class
    you created in Question 1.3 and contain the correct information as
    detailed in the csv file.

    Parameters
    ----------
    file_name
        the name of the csv file

    Returns
    -------
    robot_list
        a list of `Robot` instances

    Notes
    -----
    The csv file will be structured with the first line as headings followed
    by data on the subsequent lines. For example:

    ```csv
    name, n, joint_type, price
    Panda, 7, revolute, 40000
    Puma560, 6, revolute, 19000
    ```

    However, the order of the headings will not always be consistent (for example,
    the price heading may come before the joint_type heading).

    """

    # declare the output type
    robot_list: List[Robot] = []

    with open(file_name, 'r') as f:
        lines = f.readlines()

    headers = [h.strip() for h in lines[0].split(',')]

    for line in lines[1:]:
        values = [v.strip() for v in line.split(',')]

        row = dict(zip(headers, values))

        robot = Robot(
            name=row['name'],
            n=int(row['n']),
            joint_type=row['joint_type'],
            price=float(row['price'])
        )
        robot_list.append(robot)

    return robot_list


# --------- Question 1.5 ---------- #


def closest_robot(file_name: str, price_limit: float) -> Robot:
    """
    Finds the robot closest to a price limit

    Takes a file name of a csv file (of the same description as Question 1.4)
    and a price limit and returns a constructed Robot instance.

    - The price_limit corresponds to a price limit. The searches the csv file
      and returns a Robot instance that is closest to, or equal to, the price limit.

    Parameters
    ----------
    file_name
        the name of the csv file
    price_limit
        the price limit

    Returns
    -------
    closest_robot
        the robot closest to the price limit

    """

    robots = csv_to_robots(file_name)

    # Price filter for purchasable robots
    affordable = [r for r in robots if r.price <= price_limit]

    if affordable:
        # Returning the closest to price limit
        return max(affordable, key=lambda r: r.price)
    else:
        # If all robots exceed limit, return the cheapest robot possible
        return min(robots, key=lambda r: r.price)