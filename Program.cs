using MathNet.Numerics.LinearAlgebra;
namespace Program
{
	class Thing
	{
		Matrix<double> inputs = Matrix<double>.Build.DenseOfArray(new double[,] {
			{ 1, 2, 3, 2.5 },
			{ 2, 5, -1, 2 },
			{ -1.5, 2.7, 3.3, -0.8 }
		});
		Layer layer0 = new(4, 5, ActivationFunctions.Sigmoid);
		Layer layer1 = new(5, 2, ActivationFunctions.Sigmoid);
		public void Main()
		{
			layer0.ForwardPass(inputs);
			layer1.ForwardPass(layer0.outputs);
			Console.WriteLine(layer0.outputs);
			Console.WriteLine(layer1.outputs);
		}
	}
	static class Program
	{
		static void Main()
		{
			Thing thing = new();
			thing.Main();
		}
	}
}
