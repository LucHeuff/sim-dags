// Filter functions for edges
#let over_filter(label) = {
  let filter_edge(edge) = {
    return edge.fields().value.vertices.last() != label
  }
  return filter_edge
}

#let under_filter(label) = {
  let filter_edge(edge) = {
    return edge.fields().value.vertices.first() != label 
  }
  return filter_edge
}
