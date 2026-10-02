// dump_graphs.sc — Joern (v2.x, overflowdb) CPG 导出脚本
//
// 遍历 CPG 中每个 METHOD，抽取其 AST 子树内的节点，以及三类边：
//   AST          : 抽象语法（父子）
//   CFG          : 控制流
//   REACHING_DEF : 到达定义（作为数据流 DFG 关系）
//
// 每个方法输出一行 JSON：
//   {"f":"000123","n":[[id,label,code],...],"e":[[src,dst,"AST|CFG|REACHING_DEF"],...]}
//
// 运行: joern --script dump_graphs.sc --params cpgFile=xx.bin,outFile=xx.jsonl

import io.shiftleft.codepropertygraph.cpgloading.{CpgLoader, CpgLoaderConfig}
import io.shiftleft.semanticcpg.language._
import java.io.{BufferedWriter, FileWriter}
import scala.jdk.CollectionConverters._

@main def exec(cpgFile: String, outFile: String) = {
  val cpg = CpgLoader.loadFromOverflowDb(CpgLoaderConfig.withStorage(cpgFile))
  val edgeWanted = Set("AST", "CFG", "REACHING_DEF")
  val minNodes = 4
  val bw = new BufferedWriter(new FileWriter(outFile))

  def esc(raw: Object): String = {
    if (raw == null) return "\"\""
    val s = raw.toString
    val sb = new StringBuilder(s.length + 2)
    sb.append('"')
    var i = 0
    while (i < s.length) {
      val ch = s.charAt(i)
      ch match {
        case '"'  => sb.append("\\\"")
        case '\\' => sb.append("\\\\")
        case '\n' => sb.append("\\n")
        case '\r' => sb.append("\\r")
        case '\t' => sb.append("\\t")
        case c if c < 0x20 => // 丢弃其余控制字符
        case c    => sb.append(c)
      }
      i += 1
    }
    sb.append('"')
    sb.toString
  }

  var nMethods = 0
  var nWritten = 0
  val seenStems = new java.util.HashSet[String]()

  cpg.method.foreach { m =>
    nMethods += 1
    val base = new java.io.File(m.filename).getName
    val stem = if (base.endsWith(".c")) base.substring(0, base.length - 2) else base
    // 仅导出：数字命名源文件中的、非 <global> 的、未重复的真实函数
    if (stem.forall(_.isDigit) && m.name != "<global>" && seenStems.add(stem)) {
      val nodes = m.ast.l
      if (nodes.size >= minNodes) {
        val ids = nodes.map(_.id()).toSet

        val nodeSb = new StringBuilder("[")
        nodes.zipWithIndex.foreach { case (n, idx) =>
          if (idx > 0) nodeSb.append(",")
          val codeObj = n.propertiesMap.getOrDefault("CODE", "")
          nodeSb.append("[").append(n.id()).append(",")
            .append(esc(n.label)).append(",")
            .append(esc(codeObj)).append("]")
        }
        nodeSb.append("]")

        val edgeSb = new StringBuilder("[")
        var nEdges = 0
        val seen = new java.util.HashSet[String]()
        nodes.foreach { n =>
          n.outE("AST", "CFG", "REACHING_DEF").asScala.foreach { e =>
            val dst = e.inNode()
            val lab = e.label()
            if (edgeWanted.contains(lab) && ids.contains(dst.id())) {
              val src = e.outNode()
              val key = src.id() + "|" + dst.id() + "|" + lab
              if (seen.add(key)) {
                if (nEdges > 0) edgeSb.append(",")
                edgeSb.append("[").append(src.id()).append(",")
                  .append(dst.id()).append(",").append(esc(lab)).append("]")
                nEdges += 1
              }
            }
          }
        }
        edgeSb.append("]")

        if (nEdges > 0) {
          bw.write("{\"f\":" + esc(stem) + ",\"n\":" + nodeSb +
                   ",\"e\":" + edgeSb + "}")
          bw.newLine()
          nWritten += 1
        }
      }
    }
  }

  bw.close()
  cpg.close()
  println(s"methods scanned: $nMethods, graphs written: $nWritten -> $outFile")
}
