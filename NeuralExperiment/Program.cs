using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Accord.Math;
using Accord.Neuro;
using Accord.Neuro.Learning;

namespace NeuralExperiment
{
    internal static class Program
    {
        private static readonly Random Random = new Random();

        private static void Main(string[] args)
        {
            Network network;
            if (args.Length == 0)
                network = Learn();
            else
                using (var file = File.OpenRead(args[0]))
                {
                    network = Network.Load(file);
                }

            Calculate(network);

            using (var file = File.Create("network.dat"))
            {
                network.Save(file);
            }
        }

        private static ActivationNetwork Learn()
        {
            // Random number generation


            // Initialize network
            var network = new ActivationNetwork(new SigmoidFunction(), 6, 10, 10, 5);

            // Initialize learner (BackPropagation)
            var learner = new BackPropagationLearning(network);

            var error = 1.0;
            var sampleCount = 0;

            while (error > 0.000001)
            {
                var studentData = GenerateStudent();

                var totalScore = GetScore(studentData);

                var result = EncodeResult(totalScore);

                error = learner.Run(studentData.listingData.Concatenate(studentData.attendance), result);
                //Console.WriteLine($"Error Rate: {error.ToString(CultureInfo.InvariantCulture)}");
                sampleCount++;
            }

            Console.WriteLine($"Sample count: {sampleCount}");
            return network;
        }

        private static void Calculate(Network network)
        {
            var correct = 0;
            var wrong = 0;

            for (var i = 0; i < 1000; i++)
            {
                var student = GenerateStudent();

                Console.Write($"Student {i} - ");

                for (var j = 0; j < 5; j++)
                    Console.Write(
                        $"Q{j}: {student.listingData[j]} / ");

                Console.Write($"Attendance: {student.attendance} / ");

                var score = GetScore(student);
                var grade = GetGrade(score);

                Console.Write($"Score: {score} / True Grade: {grade} / ");

                var prediction = network.Compute(student.listingData.Concatenate(student.attendance));

                Console.WriteLine($"Predicted Grade: {GetGrade(prediction)}");

                if (GetGrade(prediction) == grade)
                    correct++;
                else
                    wrong++;
            }
            Console.WriteLine($"Correct: {correct} / Wrong {wrong}");
        }

        private static double GetScore((double[] calculationData, double[] listingData, double attendance) studentData)
        {
            var quizScore = (studentData.calculationData[0] + studentData.calculationData[1] +
                             studentData.calculationData[2] + studentData.calculationData[3] +
                             studentData.calculationData[4]) / 5;

            return quizScore * 0.95 + studentData.attendance * 0.05;
        }

        private static (double[] calculationData, double[] listingData, double attendance) GenerateStudent()
        {
            // Quiz
            var quiz = new List<double>();
            for (var j = 0; j < 6; j++)
                quiz.Add(Convert.ToDouble(Random.Next() % 100) / 100);

            // Drop the lowest score
            var quizForCalculation = DropLowest(quiz).ToArray();
            var quizForList = quiz.ToArray();

            // Attendance
            var attendance = (double) (Random.Next() % 100) / 100;

            return (quizForCalculation, quizForList, attendance);
        }

        private static double[] EncodeResult(double score)
        {
            var result = new double[5];

            for (var k = 0; k < result.Length; k++)
                result[k] = 0;

            if (score > 0.9)
                result[0] = 1.0; // A
            else if (score > 0.8)
                result[1] = 1.0; // B
            else if (score > 0.7)
                result[2] = 1.0; // C
            else if (score > 0.6)
                result[3] = 1.0; // D
            else
                result[4] = 1.0; // F

            return result;
        }

        private static string GetGrade(double score)
        {
            var encoded = EncodeResult(score);
            return GetGrade(encoded);
        }

        private static string GetGrade(double[] result)
        {
            var maxValue = result.Max();
            var maxIndex = result.ToList().IndexOf(maxValue);

            switch (maxIndex)
            {
                case 0:
                    return "A";

                case 1:
                    return "B";

                case 2:
                    return "C";

                case 3:
                    return "D";

                default:
                    return "F";
            }
        }

        private static List<double> DropLowest(List<double> scores)
        {
            scores.Sort();
            scores.RemoveAt(0);
            return scores;
        }
    }
}