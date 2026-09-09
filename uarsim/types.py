"""Core data types shared across the simulator.

Three types split the world between them, and mixing them up is the usual
source of confusion:

* :class:`Scene` — the table and what is on it. Knows nothing about language.
* :class:`Instruction` — the command. Knows nothing about the table.
* :class:`Episode` — one Scene paired with one Instruction, plus the ground
  truth, the oracle labels, and the noisy signals. This is the unit that
  datasets, policies, and metrics all operate on: one episode is one row.

A policy sees only ``Episode.signals`` and outputs an :class:`Action`;
evaluation compares the realized :class:`Outcome` against the ground truth.

Start with a table holding two mugs and a book:

    >>> from uarsim.env.task_generator import matching_ids
    >>> blue = Obj("obj_0", "mug", "blue", 0.3, 0.6, True, False, False, 0.95)
    >>> red = Obj("obj_1", "mug", "red", -0.2, 0.4, True, False, False, 0.95)
    >>> book = Obj("obj_2", "book", "red", 0.5, 0.2, True, False, False, 0.95)
    >>> scene = Scene([blue, red, book], {"laptop": (0.0, 0.8)},
    ...               fragile_in_path=False, clutter=0.25)

Ambiguity is a property of the *pair*, never of either type alone, which is why
``matching_ids`` — not Scene and not Instruction — is what resolves a command
against a table. Same table, three commands of decreasing specificity:

    >>> for cat, col in [("mug", "red"), ("mug", None), (None, None)]:
    ...     instr = Instruction("...", cat, col, "laptop")
    ...     print(cat, col, "->", matching_ids(instr, scene.objects))
    mug red -> ['obj_1']
    mug None -> ['obj_0', 'obj_1']
    None None -> ['obj_0', 'obj_1', 'obj_2']

The middle command is perfectly clear on a table that holds only one mug, so
neither type can be called "ambiguous" on its own:

    >>> instruction = Instruction("Put the mug beside the laptop.",
    ...                           ref_category="mug", ref_color=None,
    ...                           landmark="laptop")
    >>> matching_ids(instruction, [blue])
    ['obj_0']

Back on the full table it matches both mugs, so acting now is a coin flip and
the oracle prefers a cheap clarifying question:

    >>> from uarsim.config import CostConfig
    >>> from uarsim.env.oracle import compute_latents, label_episode
    >>> matches = matching_ids(instruction, scene.objects)
    >>> latents = compute_latents(scene, blue, n_matches=len(matches))
    >>> round(latents["p_correct_if_act"], 3)      # 0.5 * 0.95
    0.475
    >>> latents.update(label_episode(latents, CostConfig()))
    >>> ACTION_NAMES[Action(latents["optimal_action"])]
    'ASK'

:class:`Episode` is what bundles the pair together with everything only the
simulator knows. Asking resolves the ambiguity, so the outcome is a success:

    >>> import numpy as np
    >>> from uarsim.costs import simulate_outcome
    >>> episode = Episode(scene, instruction, "obj_0", matches, latents)
    >>> outcome, _ = simulate_outcome(Action.ASK, latents["p_correct_if_act"],
    ...                               latents["p_unsafe"],
    ...                               np.random.default_rng(0))
    >>> outcome.name
    'SUCCESS'
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Action(IntEnum):
    """The three decisions the assistance policy can output.

    Example:
        >>> int(Action.ASK), ACTION_NAMES[Action.ASK]
        (1, 'ASK')
    """

    ACT = 0     # execute autonomously
    ASK = 1     # ask a clarifying question, then execute
    DEFER = 2   # hand the task to a human


ACTION_NAMES = {Action.ACT: "ACT", Action.ASK: "ASK", Action.DEFER: "DEFER"}


class Outcome(IntEnum):
    """Realized result of an episode, used for evaluation metrics.

    Example:
        >>> Outcome(2).name
        'UNSAFE'
    """

    SUCCESS = 0        # task completed correctly (autonomously or via human)
    WRONG_OBJECT = 1   # robot acted on the wrong target
    UNSAFE = 2         # robot caused an unsafe collision


@dataclass
class Obj:
    """A single object on the table.

    ``perception_conf`` is the probability the vision model reads this object's
    identity correctly (~0.95 for familiar objects, ~0.52 for novel ones). It is
    privileged: a policy never sees it, only the noisy ``u_perception`` signal
    derived from it in :mod:`uarsim.uncertainty`. The oracle multiplies it by
    the 1/n_matches chance of picking the right candidate to get
    ``p_correct_if_act`` — misreading and mis-picking are separate failures.

    Example:
        >>> mug = Obj("obj_0", "mug", "blue", 0.3, 0.6, reachable=True,
        ...           fragile=False, novel=False, perception_conf=0.95)
        >>> mug.id, mug.perception_conf
        ('obj_0', 0.95)
    """

    id: str                 # unique within the scene, e.g. "obj_0"
    
    category: str           # "mug", "bottle", ... (see env.objects vocabularies)
    
    color: str              # "red", "blue", ...
    
    x: float                # table coords: x in [-1, 1]
    
    y: float                # y in [0, 1]; the robot base sits at BASE = (0, -1.2)
    
    reachable: bool         # if False, the oracle adds +0.30 to p_unsafe - reachable marks an object as reachable by the robot. If it's not reachable, the oracle adds 0.30 to the p_unsafe score.
    
    fragile: bool           # breakable; only matters when it sits near the
                            # reach path to the target -> Scene.fragile_in_path
    
    novel: bool             # True if its category/color is out-of-distribution - novel marks an object as being in a category or color that is not in the training data.
    
    perception_conf: float  # latent: P(model identifies this object correctly)


@dataclass
class Scene:
    """The tabletop state for one episode — the world half of the pair.

    Purely physical: no reference to the command. The same Scene can be reused
    with any number of different Instructions.

    Example:
        >>> mug = Obj("obj_0", "mug", "blue", 0.3, 0.6, True, False, False, 0.95)
        >>> scene = Scene([mug], {"laptop": (0.0, 0.8)},
        ...               fragile_in_path=False, clutter=0.25)
        >>> scene.landmarks["laptop"]
        (0.0, 0.8)
    """

    objects: list[Obj] # A list of objects on the table. Each object is an instance of the Obj class.
    
    landmarks: dict[str, tuple[float, float]]  # name -> (x, y), referenced by instructions - A dictionary that maps landmark names to their (x, y) coordinates on the table. Landmarks are objects that are referenced by instructions, such as "the laptop" or "the table". The key is the name of the landmark, and the value is a tuple of the x and y coordinates of the landmark. 
    
    fragile_in_path: bool  # reaching the target risks hitting something fragile - If True, reaching the target risks hitting something fragile. If False, reaching the target does not risk hitting something fragile. The fragile_in_path flag is used to calculate the p_unsafe score.
    
    clutter: float         # 0..1 scene density measure - A measure of how cluttered the scene is. A value of 0 means the scene is empty, and a value of 1 means the scene is completely cluttered. The clutter measure is used to calculate the p_unsafe score.

# Scene is the world; Instruction is the words. Neither knows about the other. A Scene is just objects and coordinates — you could describe it without language existing. An Instruction holds no object ids at all; it only records what the human said and which category/color words appeared.



@dataclass
class Instruction:
    """A referring expression, kept both as text and as parsed fields —
    the language half of the pair.

    Holds no object ids: which objects it refers to depends on the Scene, so
    resolving it is ``task_generator.matching_ids``' job. Leaving
    ``ref_category`` or ``ref_color`` as None is how ambiguity is introduced:
    the vaguer the reference, the more objects it can match.

    Example:
        >>> from uarsim.env.task_generator import matching_ids
        >>> mugs = [Obj(f"obj_{i}", "mug", c, 0.0, 0.5, True, False, False, 0.95)
        ...         for i, c in enumerate(("blue", "red"))]
        >>> matching_ids(Instruction("...", "mug", None, "laptop"), mugs)
        ['obj_0', 'obj_1']
        >>> matching_ids(Instruction("...", "mug", "blue", "laptop"), mugs)
        ['obj_0']
    """

    text: str
    ref_category: str | None  # category named in the command, or None ("the object")
    ref_color: str | None     # color named, or None
    landmark: str

# Episode is the pairing plus the answer key. It's a Scene, an Instruction, the resolved matches, the true target, the oracle's labels, and the noisy signals. The practical rule: one Episode is one row of your dataset — data.episode_to_row literally flattens it into one CSV line.

@dataclass
class Episode:
    """A fully-specified episode with ground truth, latents, and signals.

    A Scene and an Instruction resolved against each other, plus everything
    only the simulator knows. One episode becomes one dataset row via
    ``data.episode_to_row``.

    Example:
        >>> import numpy as np
        >>> from uarsim.config import Config
        >>> from uarsim.env.scene_generator import sample_episode
        >>> ep = sample_episode(Config(), "train", np.random.default_rng(0))
        >>> sorted(ep.signals)                                   # policy inputs
        ['u_language', 'u_ood', 'u_perception', 'u_planning', 'u_safety']
        >>> "p_unsafe" in ep.latents, "p_unsafe" in ep.signals   # privileged vs observed
        (True, False)
        >>> ep.meta["n_matching"] == len(ep.matches)
        True
    """

    scene: Scene              # the world half
    instruction: Instruction  # the language half
    true_target_id: str       # the object the human actually meant (answer key)
    matches: list[str]  # object ids matching the referring expression; >1 means ambiguous
    latents: dict = field(default_factory=dict)   # p_correct_if_act, p_unsafe, ...
    signals: dict = field(default_factory=dict)   # observed (noisy) uncertainty
    meta: dict = field(default_factory=dict)      # n_objects, mode, etc.


"""
instruction.text : 'Put the yellow bowl beside the plant.'

scene.objects:
   obj_0: blue   book    reach=True  frag=False conf=0.96
   obj_1: blue   bowl    reach=True  frag=False conf=0.95
   obj_2: blue   mug     reach=True  frag=True  conf=0.96
   obj_3: red    bottle  reach=True  frag=False conf=0.94
   obj_4: yellow bowl    reach=True  frag=True  conf=0.94   [matches]
   obj_5: yellow bowl    reach=False frag=True  conf=0.95   [matches]  <== true target

true_target_id : obj_5
matches        : ['obj_4', 'obj_5']

latents:  p_correct_if_act=0.475  p_unsafe=0.329  target_distance=0.655
          cost_act=1.339  cost_ask=1.087  cost_defer=0.300  optimal_action=DEFER
signals:  u_language=0.503  u_perception=0.093  u_planning=0.640
          u_ood=0.052  u_safety=0.338
meta:     mode=train  n_objects=6  n_matching=2  obstacle_count=3
          target_novel=0  fragile_in_path=0
"""