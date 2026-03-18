#import "functions.typ": over_filter, under_filter
#import "@preview/fletcher:0.5.8": diagram, node, edge

#set text(font: "Liberation Sans")
#set document(title: [Simple DAG derivations])

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

= Pipe

#align(center)[
#grid(
  columns: 2, 
  gutter: 5em,
  [
    #figure(
      diagram({
        nodes
        pipe
      }), caption: [Pipe diagram])
  ],
  [
    #figure(
      diagram({
          nodes
          pipe.filter(under_filter(<X>))
      }), caption: [$G_underline(X)$]
    )
  ],
)]

Using do-calculus, we have $(Y indep X)_G_underline(X)$, as $X$ is no longer connected to anything when removing all arrows pointing out of $X$. Hence, using Rule 2 of _do_-calculus, we have

$ P(y|"do"(x)) = P(y|x) $

= Fork

#align(center)[
#grid(
  columns: 3, 
  gutter: 2em,
  [
    #figure(
      diagram({
        nodes
        fork
      }), caption: [Pipe diagram])
  ],
  [
    #figure(
      diagram({
          nodes
          fork.filter(under_filter(<X>))
      }), caption: [$G_underline(X)$]
    )
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

= Collider

#align(center)[
  #grid(
    columns: 2,
    gutter: 5em,
    [
      #figure(
        diagram({
          nodes
          collider
        }), caption: [Collider diagram]
      )
    ],
    [
      #figure(
        diagram({
          nodes
          collider.filter(under_filter(<X>))
        }), caption: [$G_underline(X)$]
      )
    ]
  )
]
In this case the backdoor path through $Z$ is blocked by default due to $Z$ being a collider. This means that $(Y indep X)_G_underline(X)$ and we can apply Rule 2, which gives

$ P(y|"do"(x)) = P(y|x) $





