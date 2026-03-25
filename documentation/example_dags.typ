#import "functions.typ": over_filter, under_filter
#import "@preview/fletcher:0.5.8": diagram, node, edge

#set text(font: "Liberation Sans")
#set document(title: [Example DAG derivations])

#let indep = scale(y: 200%, x: 120%)[#sym.tack.t.double]

#let nodes = (
  node((0,0), $X$, name: <X>),
  node((2,0), $Y$, name: <Y>),
  node((1, -1), $Z$, name: <Z>)
)

#let pipe = (
  edge(<X>, <Y>, "-|>"),
  edge(<X>, <Z>, "-|>"),
  edge(<Z>, <Y>, "-|>")
)

#let fork = (
  edge(<X>, <Y>, "-|>"),
  edge(<Z>, <X>, "-|>"),
  edge(<Z>, <Y>, "-|>")
)

#let collider = (
  edge(<X>, <Y>, "-|>"),
  edge(<X>, <Z>, "-|>"),
  edge(<Y>, <Z>, "-|>")
)

#align(center)[#title()]

= Simple DAGs

== Pipe

#align(center)[
#grid(
  columns: 2, 
  gutter: 5em,
  [
    #figure(
      diagram(nodes, pipe), caption: [Pipe diagram])
  ],
  [
    #figure(
      diagram(nodes, pipe.filter(under_filter(<X>)) ), caption: [$G_underline(X)$]
    )
  ],
)]

Using do-calculus, we have $(Y indep X)_G_underline(X)$, as $X$ is no longer connected to anything when removing all arrows pointing out of $X$.
Hence, using Rule 2 of _do_-calculus, we have

$ P(y|"do"(x)) = P(y|x) $

== Fork

#align(center)[
#grid(
  columns: 3, 
  gutter: 2em,
  [
    #figure( diagram(nodes, fork), caption: [Pipe diagram])
  ],
  [
    #figure( diagram(nodes, fork.filter(under_filter(<X>))), caption: [$G_underline(X)$])
  ],
  [
    #figure(
      diagram({
        nodes
        fork.filter(over_filter(<Z>)).filter(over_filter(<X>))
      }), caption: [$G_overline(X Z)$]
    )
  ]
)]

Here in order to use Rule 2, we have instead $(Y indep X|Z)_G_underline(X)$, since there is a backdoor path through $Z$ that needs to be closed.
This means we first need to marginalise over $Z$. Then, to remove $"do"(x)$ from the second term we invoke Rule 3 given that $(X indep Z)_G_overline(X Z)$ 
since collider $Y$ closes the backdoor path. Applying these two rules gives

$ P(y|"do"(x)) &= sum_z P(y|"do"(x), z)P(z|"do"(x)) \ 
  &= sum_z P(y|x, z)P(z|"do"(x)) \
  &= sum_z P(y|x, z)P(z) \
$

== Collider

#align(center)[
  #grid(
    columns: 2,
    gutter: 5em,
    [
      #figure( diagram(nodes, collider), caption: [Collider diagram])
    ],
    [
      #figure( diagram( nodes, collider.filter(under_filter(<X>))), caption: [$G_underline(X)$])
    ]
  )
]
In this case the backdoor path through $Z$ is blocked by default due to $Z$ being a collider.
This means that $(Y indep X)_G_underline(X)$ and we can apply Rule 2, which gives

$ P(y|"do"(x)) = P(y|x) $

#pagebreak()

= Other examples

== DAG 1

#let nodes = (
  node((0,0), $X$, name: <X>),
  node((2,0), $Y$, name: <Y>),
  node((1, -1), $Z$, name: <Z>),
  node((1, 1), $W$, name: <W>)
)
#let edges = (
  edge(<X>, <Y>, "-|>"),
  edge(<Z>, <X>, "-|>"),
  edge(<Z>, <Y>, "-|>"),
  edge(<W>, <X>, "-|>"),
  edge(<W>, <Y>, "-|>")
)

#align(center)[
  #grid(
    columns: 3,
    gutter: 3em,
    [ #figure(diagram(nodes, edges), caption: [Example DAG 1])<dag1>
    ],
    [
      #figure(diagram(nodes, edges.filter(under_filter(<X>))), caption: [$G_underline(X)$])<dag1_g_under>

    ],
    [
      #figure(diagram(nodes, edges.filter(over_filter(<X>))), caption: [$G_overline(X)$])<dag1_g_over>
    ]

  )
]

Here we are looking for a set of variables $V$ such that $(Y indep X| V)_G_underline(X)$ (Rule 2).
From @dag1_g_under we see that there are still two backdoor paths through $W$ and $Z$, meaning these need to be adjusted for.
$ P(y|"do"(x)) &= sum_z P(y|"do"(x), z)P(z|"do"(x))\
&= sum_w sum_z P(y|"do"(x), z, w)P(z|"do"(x), w)P(w|"do"(x)) \
&= sum_w sum_z P(y|x, z, w)P(z|"do"(x), w)P(w|"do"(x)) \
$
Next, we want to remove the remaining $"do"(x)$ using Rule 3, needing $(Z indep X)_G_overline(X)$ and $(W indep X)_G_overline(X)$ respectively.
From @dag1_g_over we see that no additional adjustment is required, since both backdoor paths are naturally closed by the collider at $Y$.

$ P(y|"do"(x)) &= sum_w sum_z P(y|x, z, w)P(z|w)P(w)  $

As a final step, we can apply Rule 1 to $P(z|w)$ since $(Z indep W)$, $Z$ is independent of $W$ in @dag1 due to colliders at $X$ and $W$:
$ W -> Y <- Z$ blocked by collider $Y$, $ W -> X <- Z$ blocked by collider $X$ and $W -> X -> Y <- Z$ blocked by collider $Y$. Hence the estimand is
$ P(y|"do"(x)) &= sum_w sum_z P(y|x, z, w)P(z)P(w)  $








